#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"

/*
#ifdef _WIN32
#include <sysinfoapi.h>
#endif // _WIN32
*/

const auto KB = 1024;
const auto MB = 1024 * KB;
const auto WARMUPS = 10;
const auto RUNS = 100;

const int ALLOC_SIZES[] = { 4, 8, 512, 8 * KB, 8 * MB, 256 * MB, 512 * MB };
const int ALLOC_SIZES_COUNT = 7;

/*
int64_t nanos()      
{
    int64_t wintime; GetSystemTimeAsFileTime((FILETIME*)&wintime);
    return (int64_t)wintime;
}
*/

void main()
{
    double* totals = new double[ALLOC_SIZES_COUNT];
    long* devicePointers = new long[RUNS];

    void* ptr;

    // warmup
    for (int i = 0; i < WARMUPS; i++) {
        cudaMalloc(&ptr, 64);
        cudaFree(ptr);
    }
    /*
    // bench
    for (int i = 0; i < ALLOC_SIZES_COUNT; i++) {
        long elapsed = 0;
        int allocSize = ALLOC_SIZES[i];
        long ptr;
        for (int j = 0; j < RUNS; j++) {
            long start = System.nanoTime();
            ptr = JNBNative.cudaMalloc(allocSize);
            elapsed += System.nanoTime() - start;
            devicePointers[j] = ptr;
        }
        totals[i] = elapsed / NANO_PER_MS;
        for (int j = 0; j < RUNS; j++) {
            JNBNative.cudaFree(devicePointers[j]);
        }
    }

    System.out.printf("%d warmups, %d runs:%n%s", WARMUPS, RUNS, IntStream.range(0, ALLOC_SIZES.length)
        .mapToObj(i->String.format("  %d: total %.3fms, avg: %.3f%n", ALLOC_SIZES[i], totals[i], totals[i] / RUNS))
        .collect(Collectors.joining("")));
    */
}