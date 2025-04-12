// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/small_vector.hpp>
#include "common/alignment.h"
#include "common/assert.h"
#include "video_core/buffer_cache/buffer.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"    

namespace VideoCore {

std::string_view BufferTypeName(MemoryUsage type) {
    switch (type) {
    case MemoryUsage::Upload:
        return "Upload";
    case MemoryUsage::Download:
        return "Download";
    case MemoryUsage::Stream:
        return "Stream";
    case MemoryUsage::DeviceLocal:
        return "DeviceLocal";
    default:
        return "Invalid";
    }
}

[[nodiscard]] VkMemoryPropertyFlags MemoryUsagePreferredVmaFlags(MemoryUsage usage) {
    return usage != MemoryUsage::DeviceLocal ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                             : VkMemoryPropertyFlagBits{};
}

[[nodiscard]] VmaAllocationCreateFlags MemoryUsageVmaFlags(MemoryUsage usage) {
    switch (usage) {
    case MemoryUsage::Upload:
    case MemoryUsage::Stream:
        return VMA_ALLOCATION_CREATE_MAPPED_BIT |
               VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    case MemoryUsage::Download:
        return VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    case MemoryUsage::DeviceLocal:
        return {};
    }
    return {};
}

[[nodiscard]] VmaMemoryUsage MemoryUsageVma(MemoryUsage usage) {
    switch (usage) {
    case MemoryUsage::DeviceLocal:
    case MemoryUsage::Stream:
        return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    case MemoryUsage::Upload:
    case MemoryUsage::Download:
        return VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    }
    return VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
}

UniqueBuffer::UniqueBuffer(vk::Device device_, VmaAllocator allocator_)
    : device{device_}, allocator{allocator_} {}

UniqueBuffer::~UniqueBuffer() {
    if (buffer) {
        vmaDestroyBuffer(allocator, buffer, allocation);
    }
}

void UniqueBuffer::Create(const vk::BufferCreateInfo& buffer_ci, MemoryUsage usage,
                          VmaAllocationInfo* out_alloc_info) {
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage),
        .usage = MemoryUsageVma(usage),
        .requiredFlags = 0,
        .preferredFlags = MemoryUsagePreferredVmaFlags(usage),
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    const VkBufferCreateInfo buffer_ci_unsafe = static_cast<VkBufferCreateInfo>(buffer_ci);
    VkBuffer unsafe_buffer{};
    VkResult result = vmaCreateBuffer(allocator, &buffer_ci_unsafe, &alloc_ci, &unsafe_buffer,
                                      &allocation, out_alloc_info);
    ASSERT_MSG(result == VK_SUCCESS, "Failed allocating buffer with error {}",
               vk::to_string(vk::Result{result}));
    buffer = vk::Buffer{unsafe_buffer};
}

Buffer::Buffer(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_, MemoryUsage usage_,
               VAddr cpu_addr_, vk::BufferUsageFlags flags, u64 size_bytes_)
    : cpu_addr{cpu_addr_}, size_bytes{size_bytes_}, instance{&instance_}, scheduler{&scheduler_},
      usage{usage_}, buffer{instance->GetDevice(), instance->GetAllocator()} {
    // Create buffer object.
    const vk::BufferCreateInfo buffer_ci = {
        .size = size_bytes,
        .usage = flags,
    };
    VmaAllocationInfo alloc_info{};
    buffer.Create(buffer_ci, usage, &alloc_info);

    const auto device = instance->GetDevice();
    Vulkan::SetObjectName(device, Handle(), "Buffer {:#x}:{:#x}", cpu_addr, size_bytes);

    // Map it if it is host visible.
    VkMemoryPropertyFlags property_flags{};
    vmaGetAllocationMemoryProperties(instance->GetAllocator(), buffer.allocation, &property_flags);
    if (alloc_info.pMappedData) {
        mapped_data = std::span<u8>{std::bit_cast<u8*>(alloc_info.pMappedData), size_bytes};
    }
    is_coherent = property_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
}

SparseBuffer::SparseBuffer(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_, MemoryUsage usage_,
            vk::BufferUsageFlags flags, u64 size_bytes_)
    : size_bytes{size_bytes_}, instance{&instance_}, scheduler{&scheduler_},
      usage{usage_} {
    
    // Create buffer object and configure alignment.
    const auto device = instance->GetDevice();
    const vk::BufferCreateInfo buffer_ci = {
        .flags = vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = size_bytes,
        .usage = flags,
    };
    auto buffer_result = device.createBuffer(buffer_ci);
    ASSERT_MSG(buffer_result.result == vk::Result::eSuccess, "Failed creating sparse buffer with error {}",
               vk::to_string(buffer_result.result));
    buffer = buffer_result.value;
    device.getBufferMemoryRequirements(buffer, &mem_reqs);
    mem_reqs.size = mem_reqs.alignment;

    Vulkan::SetObjectName(device, buffer, "Sparse Buffer {:#x}", size_bytes);

    // Create wait fence.
    vk::FenceCreateInfo fence_ci{
        .flags = vk::FenceCreateFlagBits::eSignaled,
    };
    auto fence_result = device.createFence(fence_ci);
    ASSERT_MSG(fence_result.result == vk::Result::eSuccess, "Failed creating fence with error {}",
               vk::to_string(fence_result.result));
    fence = fence_result.value;
}

SparseBuffer::~SparseBuffer() {
    if (!buffer) {
        return;
    }
    instance->GetDevice().destroyFence(fence);
    instance->GetDevice().destroyBuffer(buffer);
    for (auto& [_, allocation] : allocations) {
        if (allocation.allocation) {
            vmaFreeMemory(instance->GetAllocator(), allocation.allocation);
        }
    }
}

bool SparseBuffer::IsInBounds(vk::DeviceAddress addr, vk::DeviceSize size) const noexcept {
    if (size == 0) {
        return true;
    }
    auto user_interval = boost::icl::interval<vk::DeviceAddress>::right_open(addr, addr + size);
    return boost::icl::contains(user_regions, user_interval);
}

void SparseBuffer::BindRegion(vk::DeviceAddress addr, vk::DeviceSize size) {
    if (size == 0) {
        return;
    }

    auto user_interval = boost::icl::interval<vk::DeviceAddress>::right_open(addr, addr + size);
    user_regions += user_interval;

    const auto aligned_start = Common::AlignUp(addr, mem_reqs.alignment);
    const auto aligned_end = Common::AlignUp(addr + size, mem_reqs.alignment);  

    VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | MemoryUsageVmaFlags(usage),
        .usage = MemoryUsageVma(usage),
        .requiredFlags = 0,
        .preferredFlags = MemoryUsagePreferredVmaFlags(usage),
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
    };

    boost::container::small_vector<vk::SparseMemoryBind, 8> binds;

    for (vk::DeviceAddress i = aligned_start; i < aligned_end; i += mem_reqs.alignment) {
        auto aligned_interval = boost::icl::interval<vk::DeviceAddress>::right_open(i, i + mem_reqs.alignment);
        if (!boost::icl::intersects(user_regions - user_interval, aligned_interval)) {
            bound_regions += aligned_interval;
            
            Allocation& allocation = allocations[i];
            ASSERT_MSG(allocation.allocation == nullptr, "Allocation already exists for address {:#x}",
                   i);
            VmaAllocationInfo alloc_info{};
            VkMemoryRequirements unsafe_mem_reqs = static_cast<VkMemoryRequirements>(mem_reqs);
            vmaAllocateMemory(instance->GetAllocator(), &unsafe_mem_reqs, &alloc_ci, &allocation.allocation, 
                              &alloc_info);
            allocation.mapped = alloc_info.pMappedData;
            allocation.device_memory = alloc_info.deviceMemory;

            binds.push_back(vk::SparseMemoryBind{
                .resourceOffset = i,
                .size = mem_reqs.alignment,
                .memory = allocation.device_memory,
                .flags = vk::SparseMemoryBindFlags{},
            });
        }
    }

    if (!binds.empty()) {
        vk::SparseBufferMemoryBindInfo bind_info{
            .buffer = buffer,
            .bindCount = static_cast<u32>(binds.size()),
            .pBinds = binds.data(),
        };
        vk::BindSparseInfo bind_sparse_info{
            .bufferBindCount = 1,
            .pBufferBinds = &bind_info,
        };
        // Todo: Make sure if this pattern for waiting is correct.
        auto result = instance->GetGraphicsQueue().bindSparse(bind_sparse_info, fence);
        ASSERT_MSG(result == vk::Result::eSuccess, "Failed binding sparse buffer with error {}",
               vk::to_string(result));
        result = instance->GetDevice().waitForFences(fence, VK_TRUE, UINT64_MAX);
        ASSERT_MSG(result == vk::Result::eSuccess, "Failed waiting for fence with error {}",
               vk::to_string(result));
        instance->GetDevice().resetFences(fence);
    }
}

void SparseBuffer::UnbindRegion(vk::DeviceAddress addr, vk::DeviceSize size) {
    if (size == 0) {
        return;
    }

    auto user_interval = boost::icl::interval<vk::DeviceAddress>::right_open(addr, addr + size);
    user_regions -= user_interval;

    const auto aligned_start = Common::AlignUp(addr, mem_reqs.alignment);
    const auto aligned_end = Common::AlignUp(addr + size, mem_reqs.alignment);

    boost::container::small_vector<vk::SparseMemoryBind, 8> binds;
    boost::container::small_vector<vk::DeviceAddress, 8> to_free;

    for (vk::DeviceAddress i = aligned_start; i < aligned_end; i += mem_reqs.alignment) {
        auto aligned_interval = boost::icl::interval<vk::DeviceAddress>::right_open(i, i + mem_reqs.alignment);
        if (boost::icl::intersects(user_regions, aligned_interval)) {
            bound_regions -= aligned_interval;
            
            auto it = allocations.find(i);
            ASSERT_MSG(it != allocations.end(), "Allocation not found for address {:#x}", i);
            Allocation& allocation = it->second;
            ASSERT_MSG(allocation.allocation != nullptr, "Allocation already freed for address {:#x}",
                   i);
            to_free.push_back(i);
            vmaFreeMemory(instance->GetAllocator(), allocation.allocation);
            allocations.erase(it);

            binds.push_back(vk::SparseMemoryBind{
                .resourceOffset = i,
                .size = mem_reqs.alignment,
                .memory = VK_NULL_HANDLE,
                .flags = vk::SparseMemoryBindFlags{},
            });
        }
    }

    if (!binds.empty()) {
        vk::SparseBufferMemoryBindInfo bind_info{
            .buffer = buffer,
            .bindCount = static_cast<u32>(binds.size()),
            .pBinds = binds.data(),
        };
        vk::BindSparseInfo bind_sparse_info{
            .bufferBindCount = 1,
            .pBufferBinds = &bind_info,
        };
        // Todo: Make sure if this pattern for waiting is correct.
        auto result = instance->GetGraphicsQueue().bindSparse(bind_sparse_info);
        ASSERT_MSG(result == vk::Result::eSuccess, "Failed binding sparse buffer with error {}",
               vk::to_string(result));
        result = instance->GetDevice().waitForFences(fence, VK_TRUE, UINT64_MAX);
        ASSERT_MSG(result == vk::Result::eSuccess, "Failed waiting for fence with error {}",
               vk::to_string(result));
        instance->GetDevice().resetFences(fence);

        for (auto i : to_free) {
            auto it = allocations.find(i);
            Allocation& allocation = it->second;
            vmaFreeMemory(instance->GetAllocator(), allocation.allocation);
            allocations.erase(it);
        }
    }
}

ImportedHostBuffer::ImportedHostBuffer(const Vulkan::Instance& instance_,
                                       Vulkan::Scheduler& scheduler_, void* cpu_addr_,
                                       u64 size_bytes_, bool with_bda, vk::BufferUsageFlags flags)
    : cpu_addr{cpu_addr_}, size_bytes{size_bytes_}, instance{&instance_}, scheduler{&scheduler_} {
    ASSERT_MSG(size_bytes > 0, "Size must be greater than 0");
    ASSERT_MSG(cpu_addr != 0, "CPU address must not be null");
    const vk::DeviceSize alignment = instance->GetExternalMemoryHostAlignment();
    ASSERT_MSG(reinterpret_cast<u64>(cpu_addr) % alignment == 0,
               "CPU address {:#x} is not aligned to {:#x}", cpu_addr, alignment);
    ASSERT_MSG(size_bytes % alignment == 0, "Size {:#x} is not aligned to {:#x}", size_bytes,
               alignment);

    vk::ImportMemoryHostPointerInfoEXT import_info{
        .pHostPointer = reinterpret_cast<void*>(cpu_addr),
    };
    vk::BufferCreateInfo buffer_ci{
        .size = size_bytes,
        .usage = flags | (with_bda ? vk::BufferUsageFlagBits::eShaderDeviceAddress : vk::BufferUsageFlags{}),
    };
    vk::MemoryAllocateInfo alloc_ci{
        .pNext = &import_info,
        .allocationSize = size_bytes,
        .memoryTypeIndex = instance->GetMemoryTypeIndex(
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
    };

    auto buffer_result = instance->GetDevice().createBuffer(buffer_ci);
    ASSERT_MSG(buffer_result.result == vk::Result::eSuccess,
               "Failed creating imported host buffer with error {}",
               vk::to_string(buffer_result.result));
    buffer = buffer_result.value;

    auto device_memory_result = instance->GetDevice().allocateMemory(alloc_ci);
    ASSERT_MSG(device_memory_result.result == vk::Result::eSuccess,
               "Failed allocating imported host memory with error {}",
               vk::to_string(device_memory_result.result));
    device_memory = device_memory_result.value;

    auto result = instance->GetDevice().bindBufferMemory(buffer, device_memory, 0);
    ASSERT_MSG(result == vk::Result::eSuccess,
               "Failed binding imported host buffer with error {}",
               vk::to_string(result));

    if (with_bda) {
        vk::BufferDeviceAddressInfo bda_info{
            .buffer = buffer,
        };
        bda_addr = instance->GetDevice().getBufferAddress(bda_info);
    }
}

ImportedHostBuffer::~ImportedHostBuffer() {
    if (!buffer) {
        return;
    }
    const auto device = instance->GetDevice();
    device.destroyBuffer(buffer);
    device.freeMemory(device_memory);
}

constexpr u64 WATCHES_INITIAL_RESERVE = 0x4000;
constexpr u64 WATCHES_RESERVE_CHUNK = 0x1000;

StreamBuffer::StreamBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                           MemoryUsage usage, u64 size_bytes)
    : Buffer{instance, scheduler, usage, 0, AllFlags, size_bytes} {
    ReserveWatches(current_watches, WATCHES_INITIAL_RESERVE);
    ReserveWatches(previous_watches, WATCHES_INITIAL_RESERVE);
    const auto device = instance.GetDevice();
    Vulkan::SetObjectName(device, Handle(), "StreamBuffer({}):{:#x}", BufferTypeName(usage),
                          size_bytes);
}

std::pair<u8*, u64> StreamBuffer::Map(u64 size, u64 alignment) {
    if (!is_coherent && usage == MemoryUsage::Stream) {
        size = Common::AlignUp(size, instance->NonCoherentAtomSize());
    }

    ASSERT(size <= this->size_bytes);
    mapped_size = size;

    if (alignment > 0) {
        offset = Common::AlignUp(offset, alignment);
    }

    if (offset + size > this->size_bytes) {
        // The buffer would overflow, save the amount of used watches and reset the state.
        invalidation_mark = current_watch_cursor;
        current_watch_cursor = 0;
        offset = 0;

        // Swap watches and reset waiting cursors.
        std::swap(previous_watches, current_watches);
        wait_cursor = 0;
        wait_bound = 0;
    }

    const u64 mapped_upper_bound = offset + size;
    WaitPendingOperations(mapped_upper_bound);
    return std::make_pair(mapped_data.data() + offset, offset);
}

void StreamBuffer::Commit() {
    if (!is_coherent) {
        if (usage == MemoryUsage::Download) {
            vmaInvalidateAllocation(instance->GetAllocator(), buffer.allocation, offset,
                                    mapped_size);
        } else {
            vmaFlushAllocation(instance->GetAllocator(), buffer.allocation, offset, mapped_size);
        }
    }

    offset += mapped_size;
    if (current_watch_cursor + 1 >= current_watches.size()) {
        // Ensure that there are enough watches.
        ReserveWatches(current_watches, WATCHES_RESERVE_CHUNK);
    }

    auto& watch = current_watches[current_watch_cursor++];
    watch.upper_bound = offset;
    watch.tick = scheduler->CurrentTick();
}

void StreamBuffer::ReserveWatches(std::vector<Watch>& watches, std::size_t grow_size) {
    watches.resize(watches.size() + grow_size);
}

void StreamBuffer::WaitPendingOperations(u64 requested_upper_bound) {
    if (!invalidation_mark) {
        return;
    }
    while (requested_upper_bound > wait_bound && wait_cursor < *invalidation_mark) {
        auto& watch = previous_watches[wait_cursor];
        wait_bound = watch.upper_bound;
        scheduler->Wait(watch.tick);
        ++wait_cursor;
    }
}

} // namespace VideoCore
