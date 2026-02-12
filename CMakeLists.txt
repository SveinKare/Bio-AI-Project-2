cmake_minimum_required(VERSION 3.14)
project(Bio-AI-Project-2 LANGUAGES CXX)

project(JsonReader VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch nlohmann/json
include(FetchContent)
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# Add your executable
add_executable(main main.cpp individual.cpp HomeCare.cpp Patient.cpp)

# Link nlohmann/json
target_link_libraries(main PRIVATE nlohmann_json::nlohmann_json)

# Platform-specific settings
if(WIN32)
    # Windows-specific settings
    target_compile_definitions(main PRIVATE WINDOWS)
elseif(APPLE)
    # macOS-specific settings
    target_compile_definitions(main PRIVATE MACOS)
endif()
