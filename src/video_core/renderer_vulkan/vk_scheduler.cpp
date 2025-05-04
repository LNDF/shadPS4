// SPDX-FileCopyrightText: Copyright 2019 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <mutex>
#include "common/assert.h"
#include "common/debug.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

std::mutex Scheduler::submit_mutex;

Scheduler::Scheduler(const Instance& instance)
    : instance{instance}, master_semaphore{instance}, command_pool{instance, &master_semaphore} {
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    AllocateWorkerCommandBuffers();

    pending_ops_waiter_thread = std::jthread(std::bind_front(&Scheduler::PendingOpWaiter, this));
}

Scheduler::~Scheduler() {
    pending_ops_waiter_thread.request_stop();
    pending_ops_waiter_thread.join();
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (is_rendering && render_state == new_state) {
        return;
    }
    EndRendering();
    is_rendering = true;
    render_state = new_state;

    const auto width =
        render_state.width != std::numeric_limits<u32>::max() ? render_state.width : 1;
    const auto height =
        render_state.height != std::numeric_limits<u32>::max() ? render_state.height : 1;

    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .offset = {0, 0},
                .extent = {width, height},
            },
        .layerCount = 1,
        .colorAttachmentCount = render_state.num_color_attachments,
        .pColorAttachments = render_state.num_color_attachments > 0
                                 ? render_state.color_attachments.data()
                                 : nullptr,
        .pDepthAttachment = render_state.has_depth ? &render_state.depth_attachment : nullptr,
        .pStencilAttachment = render_state.has_stencil ? &render_state.stencil_attachment : nullptr,
    };

    current_cmdbuf.beginRendering(rendering_info);
}

void Scheduler::EndRendering() {
    if (!is_rendering) {
        return;
    }
    is_rendering = false;
    current_cmdbuf.endRendering();
}

void Scheduler::Flush(SubmitInfo& info, bool automatic) {
    // When flushing, we only send data to the driver; no waiting is necessary.
    SubmitExecution(info, automatic);
}

void Scheduler::Finish() {
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info, true);
    Wait(presubmit_tick);
}

void Scheduler::Wait(u64 tick, bool process_pendings) {
    RENDERER_TRACE;
    if (tick >= master_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        SubmitInfo info{};
        Flush(info, true);
    }
    master_semaphore.Wait(tick);

    if (!process_pendings) {
        return;
    }

    // Only apply pending operations until the current tick.
    {
        std::scoped_lock lock{pending_ops_mutex};
        AdvancePendingOpsReadyCursor(tick);
        auto it = pending_ops_ready.begin();
        while (it != pending_ops_ready.end() && it->gpu_tick <= tick) {
            it->callback();
            ++it;
        }
        pending_ops_ready.erase(pending_ops_ready.begin(), it);
    }
}

void Scheduler::ProcessPendingOperations() {
    std::scoped_lock lock{pending_ops_mutex};
    for (auto& op : pending_ops_ready) {
        op.callback();
    }
    pending_ops_ready.clear();
}

bool Scheduler::PendingOperationsReady() const {
    std::scoped_lock lock{pending_ops_mutex};
    return !pending_ops_ready.empty();
}

void Scheduler::AllocateWorkerCommandBuffers() {
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    current_cmdbuf = command_pool.Commit();
    auto begin_result = current_cmdbuf.begin(begin_info);
    ASSERT_MSG(begin_result == vk::Result::eSuccess, "Failed to begin command buffer: {}",
               vk::to_string(begin_result));

    // Invalidate dynamic state so it gets applied to the new command buffer.
    dynamic_state.Invalidate();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

void Scheduler::CommitPendingOperations(u64 tick) {
    std::scoped_lock lock{pending_ops_mutex};
    for (auto& op : uncomitted_pending_ops) {
        pending_ops.emplace_back(std::move(op), tick);
    }
    uncomitted_pending_ops.clear();
    pending_ops_cv.notify_one();
}

void Scheduler::SubmitExecution(SubmitInfo& info, bool automatic) {
    std::scoped_lock lk{submit_mutex};
    const u64 signal_value = master_semaphore.NextTick();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    EndRendering();
    auto end_result = current_cmdbuf.end();
    ASSERT_MSG(end_result == vk::Result::eSuccess, "Failed to end command buffer: {}",
               vk::to_string(end_result));

    const vk::Semaphore timeline = master_semaphore.Handle();
    info.AddSignal(timeline, signal_value);

    static constexpr std::array<vk::PipelineStageFlags, 2> wait_stage_masks = {
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
    };

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = static_cast<u32>(info.wait_ticks.size()),
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = static_cast<u32>(info.signal_ticks.size()),
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = static_cast<u32>(info.wait_semas.size()),
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = wait_stage_masks.data(),
        .commandBufferCount = 1U,
        .pCommandBuffers = &current_cmdbuf,
        .signalSemaphoreCount = static_cast<u32>(info.signal_semas.size()),
        .pSignalSemaphores = info.signal_semas.data(),
    };

    ImGui::Core::TextureManager::Submit();
    auto submit_result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    Refresh();
    AllocateWorkerCommandBuffers();

    if (!automatic) {
        // Only commit pending operations if we explicitly flushed.
        CommitPendingOperations(signal_value);
    }
    ProcessPendingOperations();
}

void DynamicState::Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf) {
    if (dirty_state.viewports) {
        dirty_state.viewports = false;
        cmdbuf.setViewportWithCount(viewports);
    }
    if (dirty_state.scissors) {
        dirty_state.scissors = false;
        cmdbuf.setScissorWithCount(scissors);
    }
    if (dirty_state.depth_test_enabled) {
        dirty_state.depth_test_enabled = false;
        cmdbuf.setDepthTestEnable(depth_test_enabled);
    }
    if (dirty_state.depth_write_enabled) {
        dirty_state.depth_write_enabled = false;
        // Note that this must be set in a command buffer even if depth test is disabled.
        cmdbuf.setDepthWriteEnable(depth_write_enabled);
    }
    if (depth_test_enabled && dirty_state.depth_compare_op) {
        dirty_state.depth_compare_op = false;
        cmdbuf.setDepthCompareOp(depth_compare_op);
    }
    if (dirty_state.depth_bounds_test_enabled) {
        dirty_state.depth_bounds_test_enabled = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBoundsTestEnable(depth_bounds_test_enabled);
        }
    }
    if (depth_bounds_test_enabled && dirty_state.depth_bounds) {
        dirty_state.depth_bounds = false;
        if (instance.IsDepthBoundsSupported()) {
            cmdbuf.setDepthBounds(depth_bounds_min, depth_bounds_max);
        }
    }
    if (dirty_state.depth_bias_enabled) {
        dirty_state.depth_bias_enabled = false;
        cmdbuf.setDepthBiasEnable(depth_bias_enabled);
    }
    if (depth_bias_enabled && dirty_state.depth_bias) {
        dirty_state.depth_bias = false;
        cmdbuf.setDepthBias(depth_bias_constant, depth_bias_clamp, depth_bias_slope);
    }
    if (dirty_state.stencil_test_enabled) {
        dirty_state.stencil_test_enabled = false;
        cmdbuf.setStencilTestEnable(stencil_test_enabled);
    }
    if (stencil_test_enabled) {
        if (dirty_state.stencil_front_ops && dirty_state.stencil_back_ops &&
            stencil_front_ops == stencil_back_ops) {
            dirty_state.stencil_front_ops = false;
            dirty_state.stencil_back_ops = false;
            cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, stencil_front_ops.fail_op,
                                stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                stencil_front_ops.compare_op);
        } else {
            if (dirty_state.stencil_front_ops) {
                dirty_state.stencil_front_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFront, stencil_front_ops.fail_op,
                                    stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
                                    stencil_front_ops.compare_op);
            }
            if (dirty_state.stencil_back_ops) {
                dirty_state.stencil_back_ops = false;
                cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eBack, stencil_back_ops.fail_op,
                                    stencil_back_ops.pass_op, stencil_back_ops.depth_fail_op,
                                    stencil_back_ops.compare_op);
            }
        }
        if (dirty_state.stencil_front_reference && dirty_state.stencil_back_reference &&
            stencil_front_reference == stencil_back_reference) {
            dirty_state.stencil_front_reference = false;
            dirty_state.stencil_back_reference = false;
            cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_reference);
        } else {
            if (dirty_state.stencil_front_reference) {
                dirty_state.stencil_front_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_reference);
            }
            if (dirty_state.stencil_back_reference) {
                dirty_state.stencil_back_reference = false;
                cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, stencil_back_reference);
            }
        }
        if (dirty_state.stencil_front_write_mask && dirty_state.stencil_back_write_mask &&
            stencil_front_write_mask == stencil_back_write_mask) {
            dirty_state.stencil_front_write_mask = false;
            dirty_state.stencil_back_write_mask = false;
            cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                       stencil_front_write_mask);
        } else {
            if (dirty_state.stencil_front_write_mask) {
                dirty_state.stencil_front_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
                                           stencil_front_write_mask);
            }
            if (dirty_state.stencil_back_write_mask) {
                dirty_state.stencil_back_write_mask = false;
                cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, stencil_back_write_mask);
            }
        }
        if (dirty_state.stencil_front_compare_mask && dirty_state.stencil_back_compare_mask &&
            stencil_front_compare_mask == stencil_back_compare_mask) {
            dirty_state.stencil_front_compare_mask = false;
            dirty_state.stencil_back_compare_mask = false;
            cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
                                         stencil_front_compare_mask);
        } else {
            if (dirty_state.stencil_front_compare_mask) {
                dirty_state.stencil_front_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
                                             stencil_front_compare_mask);
            }
            if (dirty_state.stencil_back_compare_mask) {
                dirty_state.stencil_back_compare_mask = false;
                cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
                                             stencil_back_compare_mask);
            }
        }
    }
    if (dirty_state.primitive_restart_enable) {
        dirty_state.primitive_restart_enable = false;
        if (instance.IsPrimitiveRestartDisableSupported()) {
            cmdbuf.setPrimitiveRestartEnable(primitive_restart_enable);
        }
    }
    if (dirty_state.cull_mode) {
        dirty_state.cull_mode = false;
        cmdbuf.setCullMode(cull_mode);
    }
    if (dirty_state.front_face) {
        dirty_state.front_face = false;
        cmdbuf.setFrontFace(front_face);
    }
    if (dirty_state.blend_constants) {
        dirty_state.blend_constants = false;
        cmdbuf.setBlendConstants(blend_constants);
    }
    if (dirty_state.color_write_masks) {
        dirty_state.color_write_masks = false;
        if (instance.IsDynamicColorWriteMaskSupported()) {
            cmdbuf.setColorWriteMaskEXT(0, color_write_masks);
        }
    }
}

void Scheduler::AdvancePendingOpsReadyCursor(u64 tick) {
    // This doesn't "advance" anything. Right now, it only moves
    // elements from pending_ops to pending_ops_ready.
    auto it = pending_ops.begin();
    while (it != pending_ops.end() && it->gpu_tick <= tick) {
        pending_ops_ready.push_back(std::move(*it));
        ++it;
    }
    pending_ops.erase(pending_ops.begin(), it);
}

void Scheduler::PendingOpWaiter(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:SchedulerWaiter");
    
    auto pred = [this] {
        return !pending_ops.empty() && pending_ops.front().gpu_tick < CurrentTick();
    };

    while (!stoken.stop_requested()) {
        {
            std::unique_lock lock{pending_ops_mutex};
            Common::CondvarWait(pending_ops_cv, lock, stoken, pred);
        }
        if (stoken.stop_requested()) {
            break;
        }

        auto tick = pending_ops.front().gpu_tick;

        // We wait so that the master semaphore is refreshed
        Wait(tick, false);

        // Advance the cursor and notify
        {
            std::scoped_lock lock{pending_ops_mutex};
            AdvancePendingOpsReadyCursor(master_semaphore.KnownGpuTick());
        }
        for (auto waiter : pending_ops_waiters) {
            waiter->notify_all();
        }
    }
}

} // namespace Vulkan
