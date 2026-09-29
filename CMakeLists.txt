# -*- cmake -*-
# Example with conda:
#   cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
#         -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
#
# Options:
#   PYCUAMPCOR_CUDA   AUTO (default, build the CUDA backend if a CUDA compiler is found), ON, OFF
#   PYCUAMPCOR_DOUBLE use double precision for internal computations (default OFF)

# minimum cmake version to support CUDA and Python
cmake_minimum_required(VERSION 3.18)

# project name
project(pycuampcor LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYCUAMPCOR_CUDA AUTO CACHE STRING "Build the CUDA backend (AUTO, ON, OFF)")
set_property(CACHE PYCUAMPCOR_CUDA PROPERTY STRINGS AUTO ON OFF)
option(PYCUAMPCOR_DOUBLE "Use double precision for internal computations" OFF)

# CUDA backend
if(PYCUAMPCOR_CUDA STREQUAL "AUTO")
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(PYCUAMPCOR_WITH_CUDA ON)
    else()
        set(PYCUAMPCOR_WITH_CUDA OFF)
    endif()
elseif(PYCUAMPCOR_CUDA)
    set(PYCUAMPCOR_WITH_CUDA ON)
else()
    set(PYCUAMPCOR_WITH_CUDA OFF)
endif()

if(PYCUAMPCOR_WITH_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90)
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit 11 COMPONENTS cudart cufft REQUIRED)
    message(STATUS "Building the CUDA backend with CUDA ${CUDAToolkit_VERSION}")
else()
    message(STATUS "Not building the CUDA backend")
endif()

# CPU backend requirements
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
if(PYCUAMPCOR_DOUBLE)
    find_package(FFTW REQUIRED COMPONENTS Double)
    set(PYCUAMPCOR_FFTW_TARGET FFTW::Double)
else()
    find_package(FFTW REQUIRED COMPONENTS Float)
    set(PYCUAMPCOR_FFTW_TARGET FFTW::Float)
endif()
find_package(OpenMP COMPONENTS CXX)
if(NOT OpenMP_CXX_FOUND)
    message(WARNING "OpenMP is not found; the CPU backend will use a single thread")
endif()

# python and pybind11 requirement
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON) # Use new FindPython if available
find_package(pybind11 CONFIG REQUIRED)

# source lists
set(PYCUAMPCOR_CXX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cxx/pycuampcor)
include(${PYCUAMPCOR_CXX_DIR}/common/Sources.cmake)
include(${PYCUAMPCOR_CXX_DIR}/cpu/Sources.cmake)
include(${PYCUAMPCOR_CXX_DIR}/cuda/Sources.cmake)
list(TRANSFORM PYCUAMPCOR_COMMON_SOURCES PREPEND ${PYCUAMPCOR_CXX_DIR}/common/)
list(TRANSFORM PYCUAMPCOR_CPU_SOURCES PREPEND ${PYCUAMPCOR_CXX_DIR}/cpu/)
list(TRANSFORM PYCUAMPCOR_CUDA_SOURCES PREPEND ${PYCUAMPCOR_CXX_DIR}/cuda/)

# add a python extension module pycuampcor._<backend>, exposing class <pyclass>
# the shared code in common/ and the bindings are compiled with the backend sources
function(pycuampcor_add_backend backend pyclass)
    string(TOUPPER ${backend} BACKEND)
    set(target _${backend})
    pybind11_add_module(${target}
        ${CMAKE_CURRENT_SOURCE_DIR}/python/extensions/pycuampcor.cpp
        ${PYCUAMPCOR_COMMON_SOURCES}
        ${PYCUAMPCOR_${BACKEND}_SOURCES}
        )
    target_include_directories(${target} PRIVATE
        ${PYCUAMPCOR_CXX_DIR}/common
        ${PYCUAMPCOR_CXX_DIR}/${backend}
        )
    target_compile_definitions(${target} PRIVATE
        PYCUAMPCOR_BACKEND=${backend}
        PYCUAMPCOR_BACKEND_${BACKEND}
        PYCUAMPCOR_MODULE=${target}
        PYCUAMPCOR_CLASS=${pyclass}
        )
    if(PYCUAMPCOR_DOUBLE)
        target_compile_definitions(${target} PRIVATE CUAMPCOR_DOUBLE)
    endif()
    list(APPEND PYCUAMPCOR_TARGETS ${target})
    set(PYCUAMPCOR_TARGETS ${PYCUAMPCOR_TARGETS} PARENT_SCOPE)
endfunction()

pycuampcor_add_backend(cpu PyCPUAmpcor)
target_link_libraries(_cpu PRIVATE ${PYCUAMPCOR_FFTW_TARGET})
if(OpenMP_CXX_FOUND)
    target_link_libraries(_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()

if(PYCUAMPCOR_WITH_CUDA)
    pycuampcor_add_backend(cuda PyCuAmpcor)
    target_link_libraries(_cuda PRIVATE CUDA::cudart CUDA::cufft)
endif()

# installation
if(SKBUILD)
    # installed by scikit-build-core, which also packages python/packages/pycuampcor
    set(PYCUAMPCOR_INSTALL_DIR pycuampcor)
else()
    if(NOT DEFINED PYCUAMPCOR_INSTALL_DIR)
        set(PYCUAMPCOR_INSTALL_DIR ${Python_SITEARCH}/pycuampcor)
    endif()
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/packages/pycuampcor/
            DESTINATION ${PYCUAMPCOR_INSTALL_DIR}
            FILES_MATCHING PATTERN "*.py")
    message(STATUS "Installing pycuampcor to: ${PYCUAMPCOR_INSTALL_DIR}")
endif()

install(TARGETS ${PYCUAMPCOR_TARGETS}
        LIBRARY DESTINATION ${PYCUAMPCOR_INSTALL_DIR})
