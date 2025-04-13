// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <boost/container/small_vector.hpp>
#include "video_core/buffer_cache/host_memory_import.h"

namespace VideoCore {

void HostMemoryImport::Map(VAddr addr, u64 size) {
    auto interval = boost::icl::interval<VAddr>::right_open(addr, addr + size);
    ASSERT_MSG(!boost::icl::intersects(imported_buffers, interval),
               "Host memory import region {:#x} - {:#x} overlaps with existing region",
               addr, addr + size);
    imported_buffers += interval;
    LOG_WARNING(Common, "Host memory import region {:#x} - {:#x} mapped", addr, addr + size);
    ImportedHostBuffer buffer(instance, scheduler, reinterpret_cast<void*>(addr), size,
                              true, vk::BufferUsageFlagBits::eStorageBuffer);
    vk::DeviceAddress bda_addr = buffer.BufferDeviceAddress();
    imported_buffers_map.emplace(addr, std::move(buffer));

    // Map the buffer to the page table.
    VAddr page_start = addr >> PageShift;
    VAddr page_end = (addr + size) >> PageShift;
    
    boost::container::small_vector<vk::DeviceAddress, 16> bda_addrs;
    bda_addrs.reserve(page_end - page_start);
    for (VAddr i = page_start; i < page_end; ++i, bda_addr += PageSize) {
        bda_addrs.push_back(bda_addr);
    }

    vk::DeviceAddress bind_addr = page_start * sizeof(vk::DeviceAddress);
    vk::DeviceAddress bind_size = (page_end - page_start) * sizeof(vk::DeviceAddress);
    
    page_table.BindRegion(bind_addr, bind_size);
    page_table.WriteData<vk::DeviceAddress>(bind_addr, bda_addrs);
}

void HostMemoryImport::Unmap(VAddr addr, u64 size) {
    auto it = imported_buffers_map.find(addr);
    ASSERT_MSG(it != imported_buffers_map.end(), "Host memory import region {:#x} not found", addr);
    imported_buffers_map.erase(it);
    
    auto interval = boost::icl::interval<VAddr>::right_open(addr, addr + size);
    imported_buffers -= interval;

    // Unmap the buffer from the page table.
    VAddr page_start = addr >> PageShift;
    VAddr page_end = (addr + size) >> PageShift;

    vk::DeviceAddress bind_addr = page_start * sizeof(vk::DeviceAddress);
    vk::DeviceAddress bind_size = (page_end - page_start) * sizeof(vk::DeviceAddress);
    page_table.UnbindRegion(bind_addr, bind_size);
}

} // namespace VideoCore