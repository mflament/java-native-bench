package org.mf.bench.javanative;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/**
 * Native interface of native test methods.<br/>
 * Used to create native header files (with maven javac plugin).<br/>
 * Used to load and call native methods using JNI.
 */
public class JNBNative {

    public static final String LIBNAME = "jnb-native";

    /**
     * Test cuda malloc
     *
     * @return device pointer
     */
    public static native long cudaMalloc(long length);

    public static native void cudaFree(long ptr);

    public static void loadLibrary() {
        Runtime runtime = Runtime.getRuntime();
        try {
            runtime.loadLibrary(LIBNAME);
        } catch (UnsatisfiedLinkError e) {
            // unpack from classpath
            String libName = "jnb-native" + libExt();
            try (InputStream is = JNBNative.class.getResourceAsStream("/" + libName)) {
                if (is == null)
                    throw e;
                Files.copy(is, Path.of(libName), StandardCopyOption.REPLACE_EXISTING);
            } catch (IOException ignored) {
                throw e;
            }
            //try again, current dir is always in java.library.path
            runtime.loadLibrary(LIBNAME);
        }
    }

    private static String libExt() {
        String osName = System.getProperty("os.name", "unknown").toLowerCase();
        if (osName.startsWith("windows"))
            return ".dll";
        else if (osName.startsWith("linux"))
            return ".so";
        else if (osName.startsWith("mac"))
            return ".so"; // not really sure .. but don't really care :-)
        throw new IllegalStateException("Unknown OS " + osName);
    }

    private JNBNative() {
    }


}
