package org.mf.bench.javanative;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class BufferAccessBenchmark extends AbstractBufferBenchmark {

    private static void add(double[] c, double[] a, double[] b) {
        for (int i = 0; i < SIZE; i++) {
            c[i] = a[i] + b[i];
        }
    }

    private static void add(ByteBuffer c, ByteBuffer a, ByteBuffer b) {
        int offset;
        for (int i = 0; i < SIZE; i++) {
            offset = i * Double.BYTES;
            c.putDouble(offset, a.get(offset) + b.get(offset));
        }
    }

    private static void benchAddArray(boolean warmup) {
        final double[] a = array(), b = array(), c = array();
        long total = 0;
        int runs = warmup ? WARMUP : RUNS;
        for (int i = 0; i < runs; i++) {
            randomize(a);
            randomize(b);
            long start = System.nanoTime();
            add(c, a, b);
            total += System.nanoTime() - start;
        }

        if (!warmup) {
            System.out.printf("  %-35s : %.3fms%n", "array", total * 1e-6 / RUNS);
        }
    }

    private static void benchAddBuffer(boolean direct, ByteOrder byteOrder, boolean warmup) {
        final ByteBuffer a, b, c;
        if (direct) {
            a = directBuffer(byteOrder);
            b = directBuffer(byteOrder);
            c = directBuffer(byteOrder);
        } else {
            a = buffer(byteOrder);
            b = buffer(byteOrder);
            c = buffer(byteOrder);
        }

        long total = 0;
        int runs = warmup ? WARMUP : RUNS;
        for (int i = 0; i < runs; i++) {
            randomize(a);
            randomize(b);
            long start = System.nanoTime();
            add(c, a, b);
            total += System.nanoTime() - start;
        }
        if (!warmup) {
            String name = direct ? "direct " : "";
            name += "buffer " + byteOrder;
            System.out.printf("  %-35s : %.3fms%n", name, total * 1e-6 / RUNS);
        }
    }

    public static void main(String[] args) {
        benchAddBuffer(false, ByteOrder.nativeOrder(), true);
        benchAddBuffer(true, ByteOrder.nativeOrder(), true);
        benchAddBuffer(false, ByteOrder.BIG_ENDIAN, true);
        benchAddBuffer(true, ByteOrder.BIG_ENDIAN, true);
        benchAddArray(true);

        benchAddBuffer(false, ByteOrder.nativeOrder(), false);
        benchAddBuffer(true, ByteOrder.nativeOrder(), false);
        benchAddBuffer(false, ByteOrder.BIG_ENDIAN, false);
        benchAddBuffer(true, ByteOrder.BIG_ENDIAN, false);
        benchAddArray(false);
    }

}
