package org.mf.bench.javanative;

import org.lwjgl.BufferUtils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.util.function.Supplier;

public class DirectBufferCopyBenchmark extends AbstractBufferBenchmark {

    @FunctionalInterface
    interface BufferSupplier extends Supplier<DoubleBuffer> {
        default String name() {
            return getClass().getSimpleName();
        }
    }

    private static final int[] CHUNK_SIZES = {1, 4, 16, 128, 1024, SIZE};
    private static final BufferSupplier[] BUFFER_SUPPLIERS = {new JavaBufferSupplier(), new LwjglBufferSupplier()};

    private static double write(double[] src, int chunkSize, DoubleBuffer buffer) {
        randomize(src);
        long start = System.nanoTime();
        for (int i = 0; i < src.length; i += chunkSize) {
            chunkSize = Math.min(chunkSize, src.length - i);
            buffer.put(i, src, i, chunkSize);
        }
        return (System.nanoTime() - start) * 1e-6;
    }

    private static double read(double[] dst, int chunkSize, DoubleBuffer buffer) {
        randomize(buffer);
        long start = System.nanoTime();
        for (int i = 0; i < dst.length; i += chunkSize) {
            chunkSize = Math.min(chunkSize, dst.length - i);
            buffer.get(i, dst, i, chunkSize);
        }
        return (System.nanoTime() - start) * 1e-6;
    }

    private static void test(DoubleBuffer buffer, int chunkSize) {
        double[] expected = array();
        randomize(expected);
        double[] actual = array();
        if (expected.length != actual.length)
            throw new IllegalStateException("WTF???");
        buffer.put(0, expected, 0, expected.length);
        buffer.get(0, actual, 0, actual.length);
        for (int i = 0; i < expected.length; i++) {
            if (actual[i] != expected[i])
                throw new IllegalStateException("mismatch at offset " + i);
        }
        System.out.println(chunkSize + " ok");
    }

    private static void test() {
        for (BufferSupplier supplier : BUFFER_SUPPLIERS) {
            DoubleBuffer buffer = supplier.get();
            for (int chunkSize : CHUNK_SIZES) {
                test(buffer, chunkSize);
            }
        }
    }

    private static void bench(boolean warmup) {
        int runs = warmup ? WARMUP : RUNS;
        double[] array = array();
        for (BufferSupplier supplier : BUFFER_SUPPLIERS) {
            if (!warmup)
                System.out.printf("%s:%n", supplier.name());
            DoubleBuffer buffer = supplier.get();
            for (int chunkSize : CHUNK_SIZES) {
                double totalWrite = 0, totalRead = 0;
                for (int i = 0; i < runs; i++) {
                    totalWrite += write(array, chunkSize, buffer);
                    totalRead += read(array, chunkSize, buffer);
                }
                if (!warmup)
                    System.out.printf("  chunk size %-5d write=%.3f read=%.3f%n", chunkSize, totalWrite / runs, totalRead / runs);
            }
        }
    }

    public static void main(String[] args) {
//        test();
        bench(true);
        bench(false);
    }

    public static final class JavaBufferSupplier implements BufferSupplier {
        @Override
        public DoubleBuffer get() {
            return ByteBuffer.allocateDirect(SIZE * Double.BYTES).order(ByteOrder.nativeOrder()).asDoubleBuffer();
        }
    }

    public static final class LwjglBufferSupplier implements BufferSupplier {
        @Override
        public DoubleBuffer get() {
            return BufferUtils.createDoubleBuffer(SIZE);
        }
    }
}
