// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <boost/container/flat_map.hpp>
#include <boost/icl/interval_set.hpp>
#include "common/types.h"
#include "video_core/buffer_cache/buffer.h"

namespace VideoCore {

class HostMemoryImport {
public:
    HostMemoryImport(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_)
        : instance(instance_), scheduler(scheduler_),
          page_table(instance, scheduler, MemoryUsage::Upload, vk::BufferUsageFlagBits::eStorageBuffer, PageTableSize) {}
    
    vk::Buffer PageTable() const noexcept {
        return page_table.Handle();
    }
    
    void Map(VAddr addr, u64 size);
    void Unmap(VAddr addr, u64 size);
    static constexpr inline u64 PageSize = 16_KB;
    static constexpr inline u64 PageShift = 14;
    static constexpr inline u64 PageMask = ~((1 << PageShift) - 1);
    static constexpr inline u64 PageCount = (1UL << 40) >> PageShift; // 40 bit addreses
    static constexpr inline u64 PageTableSize = PageCount * sizeof(u64);
private:
    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    SparseBuffer page_table;
    boost::icl::interval_set<VAddr> imported_buffers;
    boost::container::flat_map<VAddr, ImportedHostBuffer> imported_buffers_map;
};

} // namespace VideoCore