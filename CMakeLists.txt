cmake_minimum_required(VERSION 3.14)
project(boa_backbones)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Torch
find_package(Torch REQUIRED)

# Add backbones library
add_library(boa_backbones STATIC
    src/backbones/gru_backbone.cpp
    src/backbones/lstm_backbone.cpp
    src/backbones/mingru_backbone.cpp
)

target_include_directories(boa_backbones PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/backbones
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(boa_backbones PUBLIC
    ${TORCH_LIBRARIES}
)

# Fetch GTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add test executable
add_executable(test_backbones
    tests/test_backbones.cpp
)

target_link_libraries(test_backbones
    boa_backbones
    ${TORCH_LIBRARIES}
    gtest_main
)

# Enable tests
enable_testing()
add_test(NAME backbones_test COMMAND test_backbones)
