package org.mf.bench.javanative;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.util.Random;

public class AbstractBufferBenchmark {
    protected static final long SEED = 12345;
    protected static final int WARMUP = 100;
    protected static final int RUNS = 1000;
    protected static final int SIZE = 50000;

    protected static double[] array() {
        return new double[SIZE];
    }

    protected static ByteBuffer buffer(ByteOrder byteOrder) {
        return ByteBuffer.allocate(SIZE * Double.BYTES).order(byteOrder);
    }

    protected static ByteBuffer directBuffer(ByteOrder byteOrder) {
        return ByteBuffer.allocateDirect(SIZE * Double.BYTES).order(byteOrder);
    }

    protected static void randomize(ByteBuffer buffer) {
        Random random = new Random(SEED);
        for (int i = 0; i < SIZE; i++) {
            buffer.putDouble(random.nextDouble());
        }
        buffer.flip();
    }

    protected static void randomize(DoubleBuffer buffer) {
        Random random = new Random(SEED);
        for (int i = 0; i < SIZE; i++) {
            buffer.put(random.nextDouble());
        }
        buffer.flip();
    }

    protected static void randomize(double[] array) {
        Random random = new Random(SEED);
        for (int i = 0; i < SIZE; i++) {
            array[i] = random.nextDouble();
        }
    }

}
