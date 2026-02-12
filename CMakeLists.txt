cmake_minimum_required(VERSION 3.14)
project(Bio-AI-Project-2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(main src/main.cpp src/individual.cpp src/ga.cpp)
