cmake_minimum_required(VERSION 3.14)
project(Bio-AI-Project-2 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release for optimised builds
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Fetch nlohmann/json
include(FetchContent)
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Find OpenMP (optional – code compiles without it)
# On macOS with Apple Clang, libomp from Homebrew needs explicit hints
if(APPLE)
    execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(HOMEBREW_LIBOMP_PREFIX)
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()
find_package(OpenMP)

# Add your executable
add_executable(main main.cpp individual.cpp HomeCare.cpp Patient.cpp GA.cpp KMeans.cpp RuinAndRepair.cpp Similarity.cpp IslandGA.cpp)

# Link nlohmann/json
target_link_libraries(main PRIVATE nlohmann_json::nlohmann_json)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found – parallel GA enabled")
else()
    message(STATUS "OpenMP not found – running single-threaded")
endif()

# Platform-specific settings
if(WIN32)
    target_compile_definitions(main PRIVATE WINDOWS)
elseif(APPLE)
    target_compile_definitions(main PRIVATE MACOS)
endif()
