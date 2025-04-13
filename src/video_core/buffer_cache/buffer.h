// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>
#include <boost/container/flat_map.hpp>
#include <boost/icl/interval_set.hpp>
#include "common/types.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"

#include <vk_mem_alloc.h>

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan


namespace VideoCore {

/// Hints and requirements for the backing memory type of a commit
enum class MemoryUsage {
    DeviceLocal, ///< Requests device local buffer.
    Upload,      ///< Requires a host visible memory type optimized for CPU to GPU uploads
    Download,    ///< Requires a host visible memory type optimized for GPU to CPU readbacks
    Stream,      ///< Requests device local host visible buffer, falling back host memory.
};

constexpr vk::BufferUsageFlags ReadFlags =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;

constexpr vk::BufferUsageFlags AllFlags =
    ReadFlags | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

struct UniqueBuffer {
    explicit UniqueBuffer(vk::Device device, VmaAllocator allocator);
    ~UniqueBuffer();

    UniqueBuffer(const UniqueBuffer&) = delete;
    UniqueBuffer& operator=(const UniqueBuffer&) = delete;

    UniqueBuffer(UniqueBuffer&& other)
        : allocator{std::exchange(other.allocator, VK_NULL_HANDLE)},
          allocation{std::exchange(other.allocation, VK_NULL_HANDLE)},
          buffer{std::exchange(other.buffer, VK_NULL_HANDLE)} {}
    UniqueBuffer& operator=(UniqueBuffer&& other) {
        buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
        allocator = std::exchange(other.allocator, VK_NULL_HANDLE);
        allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
        return *this;
    }

    void Create(const vk::BufferCreateInfo& image_ci, MemoryUsage usage,
                VmaAllocationInfo* out_alloc_info);

    operator vk::Buffer() const {
        return buffer;
    }

    vk::Device device;
    VmaAllocator allocator;
    VmaAllocation allocation;
    vk::Buffer buffer{};
};

class Buffer {
public:
    explicit Buffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                    MemoryUsage usage, VAddr cpu_addr_, vk::BufferUsageFlags flags,
                    u64 size_bytes_);

    Buffer& operator=(const Buffer&) = delete;
    Buffer(const Buffer&) = delete;

    Buffer& operator=(Buffer&&) = default;
    Buffer(Buffer&&) = default;

    /// Increases the likeliness of this being a stream buffer
    void IncreaseStreamScore(int score) noexcept {
        stream_score += score;
    }

    /// Returns the likeliness of this being a stream buffer
    [[nodiscard]] int StreamScore() const noexcept {
        return stream_score;
    }

    /// Returns true when vaddr -> vaddr+size is fully contained in the buffer
    [[nodiscard]] bool IsInBounds(VAddr addr, u64 size) const noexcept {
        return addr >= cpu_addr && addr + size <= cpu_addr + SizeBytes();
    }

    /// Returns the base CPU address of the buffer
    [[nodiscard]] VAddr CpuAddr() const noexcept {
        return cpu_addr;
    }

    /// Returns the offset relative to the given CPU address
    [[nodiscard]] u32 Offset(VAddr other_cpu_addr) const noexcept {
        return static_cast<u32>(other_cpu_addr - cpu_addr);
    }

    size_t SizeBytes() const {
        return size_bytes;
    }

    vk::Buffer Handle() const noexcept {
        return buffer;
    }

    std::optional<vk::BufferMemoryBarrier2> GetBarrier(
        vk::Flags<vk::AccessFlagBits2> dst_acess_mask, vk::PipelineStageFlagBits2 dst_stage,
        u32 offset = 0) {
        if (dst_acess_mask == access_mask && stage == dst_stage) {
            return {};
        }

        DEBUG_ASSERT(offset < size_bytes);

        auto barrier = vk::BufferMemoryBarrier2{
            .srcStageMask = stage,
            .srcAccessMask = access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_acess_mask,
            .buffer = buffer.buffer,
            .offset = offset,
            .size = size_bytes - offset,
        };
        access_mask = dst_acess_mask;
        stage = dst_stage;
        return barrier;
    }

public:
    VAddr cpu_addr = 0;
    bool is_picked{};
    bool is_coherent{};
    bool is_deleted{};
    int stream_score = 0;
    size_t size_bytes = 0;
    std::span<u8> mapped_data;
    const Vulkan::Instance* instance;
    Vulkan::Scheduler* scheduler;
    MemoryUsage usage;
    UniqueBuffer buffer;
    vk::Flags<vk::AccessFlagBits2> access_mask{
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
        vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite};
    vk::PipelineStageFlagBits2 stage{vk::PipelineStageFlagBits2::eAllCommands};
};

class SparseBuffer {
public:
    SparseBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                    MemoryUsage usage, vk::BufferUsageFlags flags, vk::DeviceSize size_bytes_);
    ~SparseBuffer();

    SparseBuffer& operator=(const SparseBuffer&) = delete;
    SparseBuffer(const SparseBuffer&) = delete;

    SparseBuffer(SparseBuffer&& other) 
        : size_bytes{std::exchange(other.size_bytes, 0)},
          instance{other.instance}, scheduler{other.scheduler}, fence{std::exchange(other.fence, VK_NULL_HANDLE)},
          usage{other.usage}, buffer{std::exchange(other.buffer, VK_NULL_HANDLE)}, mem_reqs{other.mem_reqs},
          access_mask{other.access_mask}, stage{other.stage}, allocations{std::move(other.allocations)},
          bound_regions{std::move(other.bound_regions)}, user_regions{std::move(other.user_regions)} {}
    SparseBuffer& operator=(SparseBuffer&& other) {
        size_bytes = std::exchange(other.size_bytes, 0);
        instance = other.instance;
        scheduler = other.scheduler;
        fence = std::exchange(other.fence, VK_NULL_HANDLE);
        usage = other.usage;
        buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
        mem_reqs = other.mem_reqs;
        access_mask = other.access_mask;
        stage = other.stage;
        allocations = std::move(other.allocations);
        bound_regions = std::move(other.bound_regions);
        user_regions = std::move(other.user_regions);
        return *this;
    }

    /// Returns true when addr -> addr+size is fully contained in the buffer and it is bound
    [[nodiscard]] bool IsInBounds(vk::DeviceAddress addr, vk::DeviceSize size) const noexcept;

    size_t SizeBytes() const {
        return size_bytes;
    }

    vk::Buffer Handle() const noexcept {
        return buffer;
    }

    std::optional<vk::BufferMemoryBarrier2> GetBarrier(
        vk::Flags<vk::AccessFlagBits2> dst_acess_mask, vk::PipelineStageFlagBits2 dst_stage,
        u32 offset = 0) {
        if (dst_acess_mask == access_mask && stage == dst_stage) {
            return {};
        }

        DEBUG_ASSERT(offset < size_bytes);

        auto barrier = vk::BufferMemoryBarrier2{
            .srcStageMask = stage,
            .srcAccessMask = access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_acess_mask,
            .buffer = buffer,
            .offset = offset,
            .size = size_bytes - offset,
        };
        access_mask = dst_acess_mask;
        stage = dst_stage;
        return barrier;
    }

    // Binds a region of the buffer. Alignment is handled internally.
    void BindRegion(vk::DeviceAddress addr, vk::DeviceSize size);

    // Unbinds a region of the buffer. Alignment is handled internally.
    void UnbindRegion(vk::DeviceAddress addr, vk::DeviceSize size);

    // Write data to the buffer. Alignment is handled internally.
    template <typename T>
    void WriteData(vk::DeviceAddress addr, std::span<const T> data) {
        const size_t total_size = data.size_bytes();
        const u8* p = reinterpret_cast<const u8*>(data.data());

        vk::DeviceAddress start = Common::AlignDown(addr, mem_reqs.alignment);
        vk::DeviceAddress end = Common::AlignUp(addr + total_size, mem_reqs.alignment);
        vk::DeviceAddress final = addr + total_size;

        auto it = allocations.find(start);

        for (vk::DeviceAddress region = start; region < end; region += mem_reqs.alignment, ++it) {
            ASSERT_MSG(it != allocations.end()  && it->first == region, "Write to non bound address {:#x}", region);
            Allocation& allocation = it->second;
            ASSERT_MSG(allocation.mapped != nullptr, "Write to non host visible address {:#x}",
                       region);

            vk::DeviceAddress region_start = region;
            vk::DeviceAddress region_end = region + mem_reqs.alignment;

            vk::DeviceAddress copy_start = std::max(addr, region_start);
            vk::DeviceAddress copy_end = std::min(final, region_end);

            size_t offset = copy_start - region_start;
            size_t copy_size = copy_end - copy_start;

            u8* mapped = static_cast<u8*>(allocation.mapped);
            std::memcpy(mapped + offset, p, copy_size);
            p += copy_size;
        }
    }

    // Read data from the buffer. Alignment is handled internally.
    template <typename T>
    void ReadData(vk::DeviceAddress addr, std::span<T> data) {
        const size_t total_size = data.size_bytes();
        u8* p = reinterpret_cast<u8*>(std::addressof(data[0]));

        vk::DeviceAddress start = Common::AlignDown(addr, mem_reqs.alignment);
        vk::DeviceAddress end = Common::AlignUp(addr + total_size, mem_reqs.alignment);
        vk::DeviceAddress final = addr + total_size;

        auto it = allocations.find(start);

        for (vk::DeviceAddress region = start; region < end; region += mem_reqs.alignment, ++it) {
            ASSERT_MSG(it != allocations.end()  && it->first == region, "Read from non bound address {:#x}", region);
            Allocation& allocation = it->second;
            ASSERT_MSG(allocation.mapped != nullptr, "Read from non host visible address {:#x}",
                       region);

            vk::DeviceAddress region_start = region;
            vk::DeviceAddress region_end = region + mem_reqs.alignment;

            vk::DeviceAddress copy_start = std::max(addr, region_start);
            vk::DeviceAddress copy_end = std::min(final, region_end);

            size_t offset = copy_start - region_start;
            size_t copy_size = copy_end - copy_start;

            u8* mapped = static_cast<u8*>(allocation.mapped);
            std::memcpy(p, mapped + offset, copy_size);
            p += copy_size;
        }
    }
private:
    struct Allocation {
        VmaAllocation allocation;
        vk::DeviceMemory device_memory;
        void* mapped;
    };

    size_t size_bytes = 0;
    const Vulkan::Instance* instance;
    Vulkan::Scheduler* scheduler;
    vk::Fence fence;
    MemoryUsage usage;
    vk::Buffer buffer;
    vk::MemoryRequirements mem_reqs{};
    vk::Flags<vk::AccessFlagBits2> access_mask{
        vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite |
        vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite};
        vk::PipelineStageFlagBits2 stage{vk::PipelineStageFlagBits2::eAllCommands};
    boost::container::flat_map<vk::DeviceAddress, Allocation> allocations;
    boost::icl::interval_set<vk::DeviceAddress> bound_regions;
    boost::icl::interval_set<vk::DeviceAddress> user_regions;
};

class ImportedHostBuffer {
public:
    ImportedHostBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                        void* cpu_addr_, u64 size_bytes_, bool with_bda, vk::BufferUsageFlags flags);
    ~ImportedHostBuffer();


    ImportedHostBuffer& operator=(const ImportedHostBuffer&) = delete;
    ImportedHostBuffer(const ImportedHostBuffer&) = delete;

    ImportedHostBuffer(ImportedHostBuffer&& other)
        : size_bytes{std::exchange(other.size_bytes, 0)},
          cpu_addr{std::exchange(other.cpu_addr, nullptr)},
          bda_addr{std::exchange(other.bda_addr, 0)},
          instance{other.instance}, scheduler{other.scheduler},
          buffer{std::exchange(other.buffer, VK_NULL_HANDLE)},
          device_memory{std::exchange(other.device_memory, VK_NULL_HANDLE)} {}
    ImportedHostBuffer& operator=(ImportedHostBuffer&& other) {
        size_bytes = std::exchange(other.size_bytes, 0);
        cpu_addr = std::exchange(other.cpu_addr, nullptr);
        bda_addr = std::exchange(other.bda_addr, false);
        instance = other.instance;
        scheduler = other.scheduler;
        buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
        device_memory = std::exchange(other.device_memory, VK_NULL_HANDLE);
        return *this;
    }

    /// Returns the base CPU address of the buffer
    void* CpuAddr() const noexcept {
        return cpu_addr;
    }

    // Returns the handle to the Vulkan buffer
    vk::Buffer Handle() const noexcept {
        return buffer;
    }

    // Returns the size of the buffer in bytes
    size_t SizeBytes() const noexcept {
        return size_bytes;
    }

    // Returns the Buffer Device Address of the buffer
    vk::DeviceAddress BufferDeviceAddress() const noexcept {
        ASSERT_MSG(bda_addr != 0, "Can't get BDA from a non BDA buffer");
        return bda_addr;
    }
private:
    size_t size_bytes = 0;
    void* cpu_addr = 0;
    vk::DeviceAddress bda_addr = 0;
    const Vulkan::Instance* instance;
    Vulkan::Scheduler* scheduler;
    vk::Buffer buffer;
    vk::DeviceMemory device_memory;
};

class StreamBuffer : public Buffer {
public:
    explicit StreamBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                          MemoryUsage usage, u64 size_bytes_);

    /// Reserves a region of memory from the stream buffer.
    std::pair<u8*, u64> Map(u64 size, u64 alignment = 0);

    /// Ensures that reserved bytes of memory are available to the GPU.
    void Commit();

    /// Maps and commits a memory region with user provided data
    u64 Copy(auto src, size_t size, size_t alignment = 0) {
        const auto [data, offset] = Map(size, alignment);
        std::memcpy(data, reinterpret_cast<const void*>(src), size);
        Commit();
        return offset;
    }

    u64 GetFreeSize() const {
        return size_bytes - offset - mapped_size;
    }

private:
    struct Watch {
        u64 tick{};
        u64 upper_bound{};
    };

    /// Increases the amount of watches available.
    void ReserveWatches(std::vector<Watch>& watches, std::size_t grow_size);

    /// Waits pending watches until requested upper bound.
    void WaitPendingOperations(u64 requested_upper_bound);

private:
    u64 offset{};
    u64 mapped_size{};
    std::vector<Watch> current_watches;
    std::size_t current_watch_cursor{};
    std::optional<size_t> invalidation_mark;
    std::vector<Watch> previous_watches;
    std::size_t wait_cursor{};
    u64 wait_bound{};
};

} // namespace VideoCore
