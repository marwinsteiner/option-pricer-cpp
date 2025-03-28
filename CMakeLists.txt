cmake_minimum_required(VERSION 3.14)
project(option_pricer_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Direct include of Eigen (no need for find_package)
include_directories("C:/Users/marwi/CLionProjects/packages/eigen-3.4.0")

# Enable OpenMP for MinGW
if(MINGW)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
else()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# Add library
add_library(vanilla_options
        src/models/black_scholes.cpp
        src/pricers/black_scholes_pricer.cpp
        src/models/normal.cpp
        src/pricers/normal_pricer.cpp
        src/models/stochastic_vol_model.cpp
        src/models/log_normal_sv.cpp
        src/models/heston_sv.cpp
)

target_include_directories(vanilla_options
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Optimization flags for MinGW
if(MINGW)
    target_compile_options(vanilla_options PRIVATE
            -O3
            -march=native
            -ffast-math
    )
endif()

# Add examples
add_subdirectory(examples)