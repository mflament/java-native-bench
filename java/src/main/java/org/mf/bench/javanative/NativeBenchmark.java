package org.mf.bench.javanative;

import com.sun.jna.Library;
import com.sun.jna.Native;

import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class NativeBenchmark {

    private static final double NANO_PER_MS = TimeUnit.MILLISECONDS.toNanos(1);
    private static final int KB = 1024;
    private static final int MB = 1024 * KB;

    private static final int WARMUPS = 10;
    private static final int RUNS = 1000;
    private static final int[] ALLOC_SIZES = {4, 8, 512, 8 * KB, 8 * MB, 256 * MB};

    public NativeBenchmark() {
    }

    public static void bench(CuMAPI api) {
        double[] totals = new double[ALLOC_SIZES.length];
        long[] devicePointers = new long[RUNS];

        // test
        long ptr = api.cudaMalloc(64);
        if (ptr == 0) throw new IllegalStateException("cudaMalloc failed");
        api.cudaFree(ptr);

        // warmup
        for (int i = 0; i < WARMUPS; i++) {
            api.cudaFree(api.cudaMalloc(64));
        }

        // bench
        for (int i = 0; i < ALLOC_SIZES.length; i++) {
            long elapsed = 0;
            int allocSize = ALLOC_SIZES[i];
            for (int j = 0; j < RUNS; j++) {
                final long start = System.nanoTime();
                ptr = api.cudaMalloc(allocSize);
                elapsed += System.nanoTime() - start;
                devicePointers[j] = ptr;
            }
            totals[i] = elapsed / NANO_PER_MS;
            for (int j = 0; j < RUNS; j++) {
                api.cudaFree(devicePointers[j]);
            }
        }

        System.out.printf("%s : %d warmups, %d runs:%n%s", api.name(), WARMUPS, RUNS, IntStream.range(0, ALLOC_SIZES.length)
                .mapToObj(i -> String.format("  %d: total %.3fms, avg: %.3fms%n", ALLOC_SIZES[i], totals[i], totals[i] / RUNS))
                .collect(Collectors.joining("")));
    }

    public static void main(String[] args) {
        CuMAPI api;
        String arg = args.length > 0 ? args[0] : null;
        if (arg != null && arg.equals("jna"))
            api = jnaApi();
        else
            api = jniApi();
        bench(api);

        if (arg != null && arg.equals("both")) {
            api = jnaApi();
            bench(api);
        }
    }

    // a Cuda ultra minimalist API
    interface CuMAPI {
        String name();

        long cudaMalloc(long size);

        void cudaFree(long ptr);
    }

    private static CuMAPI jniApi() {
        JNBNative.loadLibrary();
        return new CuMAPI() {
            @Override
            public String name() {
                return "jni";
            }

            @Override
            public long cudaMalloc(long size) {
                return JNBNative.cudaMalloc(size);
            }

            @Override
            public void cudaFree(long ptr) {
                JNBNative.cudaFree(ptr);
            }
        };
    }

    private static CuMAPI jnaApi() {
        CudaLibrary library = Native.load("cudart64_110.dll", CudaLibrary.class);
        return new CuMAPI() {
            @Override
            public String name() {
                return "jna";
            }

            @Override
            public long cudaMalloc(long size) {
                long[] ptr = new long[1];
                library.cudaMalloc(ptr, size);
                return ptr[0];
            }

            @Override
            public void cudaFree(long ptr) {
                library.cudaFree(ptr);
            }
        };
    }

    interface CudaLibrary extends Library {
        int cudaMalloc(long[] ptr, long size);

        int cudaFree(long ptr);
    }

}