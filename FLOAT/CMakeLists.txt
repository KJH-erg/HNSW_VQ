cmake_minimum_required (VERSION 3.10)

project(ExRaBitQ LANGUAGES CXX)

set(CXX_STANDARD 17)


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
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall -Wextra -fopenmp -fpic -fexceptions -lrt -ftree-vectorize -march=native")

add_subdirectory(src)