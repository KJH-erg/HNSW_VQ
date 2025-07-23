cmake_minimum_required(VERSION 3.10)
project(ExRaBitQ LANGUAGES CXX)

set(CXX_STANDARD 17)

# Add FAISS include and lib paths
set(FAISS_ROOT "${CMAKE_SOURCE_DIR}/../../third/faiss")
set(FAISS_INCLUDE_DIR "${FAISS_ROOT}")
set(FAISS_LIB_DIR "${FAISS_ROOT}/build/faiss")

# === Include and link FAISS ===
include_directories(${FAISS_INCLUDE_DIR})
link_directories(${FAISS_LIB_DIR})

# === Add compile flags (optional tweaks) ===
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)



# Optional: If you're using FAISS with OpenMP or BLAS, find and link those too
find_package(Threads REQUIRED)

# AVX512 required
include(CheckCXXSourceCompiles)
set(AVX2_CXX_CODE "#include <immintrin.h>
int main() {
#if __AVX512F__
return 0;
#else
#error \"AVX512F is not supported\"
#endif
}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Ofast -lrt -march=native -fopenmp -ftree-vectorize -fexceptions -mavx2 -mfma")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -g -O0 -lrt -march=native -fopenmp -ftree-vectorize -fexceptions -mavx2 -mfma")

add_subdirectory(src)