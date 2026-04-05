cmake_minimum_required(VERSION 3.14)
project(transformer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type defaults to Release (optimized)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0 -fsanitize=address")

# Find OpenMP (optional — used for parallel matmul)
find_package(OpenMP)

# Find nlohmann_json
find_package(nlohmann_json REQUIRED)

# ── Core library ──────────────────────────────────────────
# All ops compiled into a static library so both the main
# binary and tests can link against it without recompiling.
add_library(transformer_lib STATIC
    src/ops.cpp
    src/attention.cpp
    src/transformer.cpp
    src/loader.cpp
)

target_include_directories(transformer_lib PUBLIC include)
target_link_libraries(transformer_lib PUBLIC nlohmann_json::nlohmann_json)

if(OpenMP_CXX_FOUND)
    target_link_libraries(transformer_lib PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(transformer_lib PUBLIC USE_OPENMP)
endif()

# ── Main inference binary ─────────────────────────────────
add_executable(transformer src/main.cpp)
target_link_libraries(transformer PRIVATE transformer_lib)

# ── Tests ─────────────────────────────────────────────────
add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE transformer_lib)

# ── Benchmark ─────────────────────────────────────────────
add_executable(bench_matmul bench/bench_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE transformer_lib)