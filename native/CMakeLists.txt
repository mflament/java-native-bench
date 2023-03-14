cmake_minimum_required(VERSION 3.9)
set(CMAKE_GENERATOR_PLATFORM "x64")

find_package(JNI)
if (!JNI_FOUND)
    message (FATAL_ERROR "JNI not found")
endif()

find_package(CUDAToolkit)
if (!CUDAToolkit_FOUND)
    message (FATAL_ERROR "CUDA Toolkit not found")
endif()

project(jnb-native)

add_library(jnb-native SHARED src/main/native/jnb_native.cpp)
target_link_libraries(jnb-native CUDA::cudart)

add_executable(benchmark src/main/native/main.cpp)
target_link_libraries(benchmark CUDA::cudart)

include_directories("target/generated-sources/native/include" ${JNI_INCLUDE_DIRS} ${CUDAToolkit_INCLUDE_DIRS})
