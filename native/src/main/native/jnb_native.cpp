#include <stdio.h>
#include <jni.h>
#include "cuda_runtime.h"
#include "org_mf_bench_javanative_JNBNative.h"

/*
 * Class:     org_mf_bench_javanative_JNBNative
 * Method:    cudaMalloc
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_org_mf_bench_javanative_JNBNative_cudaMalloc(JNIEnv* env, jclass, jlong size)
{
    void* ptr = NULL;
    cudaError err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess)
        return 0;
    return (jlong)ptr;
}

/*
 * Class:     org_mf_bench_javanative_JNBNative
 * Method:    cudaFree
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_org_mf_bench_javanative_JNBNative_cudaFree(JNIEnv* env, jclass, jlong ptr) 
{
    cudaFree((void*)ptr);
}
