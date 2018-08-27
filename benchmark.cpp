/**
 * \file
 * Copyright 2018 Jonas Schenke, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * alpaka is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * alpaka is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with alpaka.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */

//! @todo: in extra repo mit readme

#include "alpakaConfig.hpp"
#include "appHelper.hpp"
#include "kernel.hpp"
#include <alpaka/alpaka.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

// accelerator defines
using TAccelerator = GpuCudaRt;

using DevAcc = TAccelerator::DevAcc;
using DevHost = TAccelerator::DevHost;
using QueueAcc = TAccelerator::Stream;
using Accelerator = TAccelerator::Acc;
using PltfAcc = TAccelerator::PltfAcc;
using PltfHost = TAccelerator::PltfHost;
using MaxBlockSize = TAccelerator::MaxBlockSize;

using Clock = std::chrono::high_resolution_clock;
using microseconds = std::chrono::microseconds;

//-----------------------------------------------------------------------------
//! Synchronizes the device and measure the time.
//!
//! \param acc The accelerator device.
Clock::time_point getTimeEvent(DevAcc acc)
{
    alpaka::wait::wait(acc);
    return Clock::now();
}

//-----------------------------------------------------------------------------
//! Returns the time difference between two points in milliseconds.
//!
//! \param t1 The first time point.
//! \param t1 The second time point.
double timediff(Clock::time_point t1, Clock::time_point t2)
{
    return std::max(std::chrono::duration_cast<microseconds>(t2 - t1),
                    std::chrono::duration_cast<microseconds>(t1 - t2))
               .count() /
           1000.0;
}

//-----------------------------------------------------------------------------
//! Reduction benchmark.
//!
//! \tparam T The data type.
//! \tparam TAcc The accelerator.
//! \tparam TRuns The number of runs.
//! \tparam TBlockSize The block size.
//!
//! \param n The problem size.
//! \param dev The accelerator device number.
template <typename T,
          typename TAcc,
          unsigned int TRuns,
          unsigned int TBlockSize>
void reduce(size_t n, int dev)
{
    // get queue and other device specific properties
    DevAcc devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(dev));
    DevHost devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

    QueueAcc queue(devAcc);
    auto prop = alpaka::acc::getAccDevProps<Accelerator, DevAcc>(devAcc);
    Clock::time_point timeStart, timeEnd;

    unsigned int blockCount = prop.m_multiProcessorCount;
    unsigned int maxBlockCount =
        (((n + 1) / 2) - 1) / TBlockSize + 1; // ceil(ceil(n/2.0)/TBlockSize)

    // allocate memory
    auto hostMemory = alpaka::mem::view::ViewPlainPtr<DevHost, T, Dim, Idx>(
        new T[n], devHost, n);
    alpaka::mem::buf::Buf<DevAcc, T, Dim, Extent> sourceDeviceMemory =
        alpaka::mem::buf::alloc<T, Idx>(devAcc, n);

    T *nativeHostMemory = alpaka::mem::view::getPtrNative(hostMemory);

    // fill array with data
    for (unsigned int i = 0; i < n; i++)
        nativeHostMemory[i] = static_cast<T>(i + 1);

    // upload data
    alpaka::mem::view::copy(queue, sourceDeviceMemory, hostMemory, n);

    unsigned int i = 0;
    // -- GRIDSIZE LOOP --
    do
    {
        blockCount <<=
            1; // starting with 2*prop.m_multiProcessorCount blocks per grid
        if (blockCount > maxBlockCount)
            blockCount = maxBlockCount;

        // print properties
        std::cout << " " << std::setw(3) << i++ << ",\t" << dev << ",\t"
                  << alpaka::acc::getAccName<DevAcc>() << ",\t" << n << ",\t\t"
                  << std::setw(3) << prop.m_multiProcessorCount << ",\t\t"
                  << std::setw(9) << blockCount << ",\t" << std::setw(9)
                  << blockCount / prop.m_multiProcessorCount << ",\t\t"
                  << std::setw(7) << maxBlockCount << ",\t" << TBlockSize
                  << ",\t\t" << TRuns;

        // allocate configuration specific memory
        alpaka::mem::buf::Buf<DevAcc, T, Dim, Extent> destinationDeviceMemory =
            alpaka::mem::buf::alloc<T, Idx>(
                devAcc, static_cast<Extent>(blockCount));

        // begin time measurement
        float milliseconds = 0;
        float minMs = std::numeric_limits<float>::max();

        auto addFn = [] ALPAKA_FN_ACC(T a, T b) -> T { return a + b; };

        // -- REPETITIONS --
        for (unsigned int r = 0; r < TRuns; ++r)
        {
            // sync devices and start time measurement
            timeStart = getTimeEvent(devAcc);

            // create kernels with their workdivs
            ReduceKernel<TBlockSize, T, decltype(addFn)> kernel1, kernel2;
            WorkDiv workDiv1{ static_cast<Extent>(blockCount),
                              static_cast<Extent>(TBlockSize),
                              static_cast<Extent>(1) };
            WorkDiv workDiv2{ static_cast<Extent>(1),
                              static_cast<Extent>(TBlockSize),
                              static_cast<Extent>(1) };

            // execute first kernel
            auto const exec1(alpaka::kernel::createTaskExec<Accelerator>(
                workDiv1,
                kernel1,
                alpaka::mem::view::getPtrNative(sourceDeviceMemory),
                alpaka::mem::view::getPtrNative(destinationDeviceMemory),
                n,
                addFn));

            // reduce the last block
            auto const exec2(alpaka::kernel::createTaskExec<Accelerator>(
                workDiv2,
                kernel2,
                alpaka::mem::view::getPtrNative(destinationDeviceMemory),
                alpaka::mem::view::getPtrNative(destinationDeviceMemory),
                blockCount,
                addFn));

            // enqueue both kernels
            alpaka::queue::enqueue(queue, exec1);
            alpaka::queue::enqueue(queue, exec2);

            // stop time measurement
            timeEnd = getTimeEvent(devAcc);
            milliseconds = timediff(timeStart, timeEnd);

            if (milliseconds < minMs)
                minMs = milliseconds;
        }

        // download result from GPU
        T resultGpuHost;
        auto resultGpuDevice =
            alpaka::mem::view::ViewPlainPtr<DevHost, T, Dim, Idx>(
                &resultGpuHost, devHost, (Extent)TBlockSize);

        alpaka::mem::view::copy(
            queue, resultGpuDevice, destinationDeviceMemory, 1);

        // check result
        T expectedResult = static_cast<T>(n / 2.0 * (n + 1.0));
        if (resultGpuHost != expectedResult)
        {
            std::cerr << "\n\n"
                      << resultGpuHost << " != " << expectedResult << "\n";
            throw std::runtime_error("RESULT MISMATCH");
        }
        std::cout << ",\t" << std::setw(7) << minMs << " ms"
                  << ",\t" << n * sizeof(T) / minMs * 1e-6
                  << "\n";

    } while (blockCount < maxBlockCount);
}

int main(int argc, const char **argv)
{
    // define constants
    static constexpr unsigned int REPETITIONS = 5;
    static constexpr unsigned int BLOCKSIZE64 =
        getMaxBlockSize<TAccelerator, 64>();
    static constexpr unsigned int BLOCKSIZE128 =
        getMaxBlockSize<TAccelerator, 128>();
    static constexpr unsigned int BLOCKSIZE256 =
        getMaxBlockSize<TAccelerator, 256>();
    static constexpr unsigned int BLOCKSIZE512 =
        getMaxBlockSize<TAccelerator, 512>();
    static constexpr unsigned int BLOCKSIZE1024 =
        getMaxBlockSize<TAccelerator, 1024>();
    using DATA_TYPE = unsigned;

    // select benchmarking parameters
    const int dev = 0;
    unsigned int n1 = 0;
    unsigned int n2 = 0;
    if (argc >= 2)
        n1 = atoi(argv[1]);
    if (n1 < 2)
        n1 = 1 << 28;
    if (argc == 3) // range
        n2 = atoi(argv[2]);
    if (n2 < n1)
        n2 = n1;

    // print table header
    print_header("reduction-grid", n1, n2);

    // start the benchmark
    try
    {
        for (unsigned n = n1; n <= n2; n <<= 1)
        {
            // reduce
            reduce<DATA_TYPE, Accelerator, REPETITIONS, BLOCKSIZE64>(n, dev);
            reduce<DATA_TYPE, Accelerator, REPETITIONS, BLOCKSIZE128>(n, dev);
            reduce<DATA_TYPE, Accelerator, REPETITIONS, BLOCKSIZE256>(n, dev);
            reduce<DATA_TYPE, Accelerator, REPETITIONS, BLOCKSIZE512>(n, dev);
            reduce<DATA_TYPE, Accelerator, REPETITIONS, BLOCKSIZE1024>(n, dev);
        }
    }
    catch (std::runtime_error e)
    {
        std::cerr << e.what() << "\n";
        alpaka::dev::reset(alpaka::pltf::getDevByIdx<PltfAcc>(dev));
        return 1;
    }
    alpaka::dev::reset(alpaka::pltf::getDevByIdx<PltfAcc>(dev));

    return 0;
}
