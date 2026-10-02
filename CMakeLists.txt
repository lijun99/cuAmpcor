# -*- cmake -*-
# Example with conda:
#   cmake -S . -B build -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
#         -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
#
# Options:
#   PYCUAMPCOR_CUDA   AUTO (default, build the CUDA backend if a CUDA compiler is found), ON, OFF
#   PYCUAMPCOR_HDF5   AUTO (default, read images from HDF5 datasets if HDF5 and zlib are found), ON, OFF
#   PYCUAMPCOR_DOUBLE use double precision for internal computations (default OFF)
#   PYCUAMPCOR_BUILD_TESTS build the C++ unit tests, run with ctest (default OFF)

# minimum cmake version to support CUDA and Python
cmake_minimum_required(VERSION 3.18)

# project name
project(pycuampcor VERSION 2.1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYCUAMPCOR_CUDA AUTO CACHE STRING "Build the CUDA backend (AUTO, ON, OFF)")
set_property(CACHE PYCUAMPCOR_CUDA PROPERTY STRINGS AUTO ON OFF)
set(PYCUAMPCOR_HDF5 AUTO CACHE STRING "Read images from HDF5 datasets (AUTO, ON, OFF)")
set_property(CACHE PYCUAMPCOR_HDF5 PROPERTY STRINGS AUTO ON OFF)
option(PYCUAMPCOR_DOUBLE "Use double precision for internal computations" OFF)
option(PYCUAMPCOR_BUILD_TESTS "Build the C++ unit tests (requires GoogleTest)" OFF)

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
        if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
            # all major architectures supported by the CUDA toolkit in use
            set(CMAKE_CUDA_ARCHITECTURES all-major)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
        endif()
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

# HDF5 (C library) and zlib, for reading images from HDF5 datasets
set(PYCUAMPCOR_WITH_HDF5 OFF)
if(PYCUAMPCOR_HDF5 STREQUAL "AUTO" OR PYCUAMPCOR_HDF5)
    if(PYCUAMPCOR_HDF5 STREQUAL "AUTO")
        find_package(HDF5 COMPONENTS C)
        find_package(ZLIB)
    else()
        find_package(HDF5 REQUIRED COMPONENTS C)
        find_package(ZLIB REQUIRED)
    endif()
    if(HDF5_FOUND AND ZLIB_FOUND)
        if(HDF5_VERSION VERSION_LESS 1.14)
            message(STATUS "HDF5 ${HDF5_VERSION} < 1.14: HDF5 datasets are read with the library only")
        endif()
        if(NOT TARGET HDF5::HDF5)
            # cmake < 3.19
            add_library(HDF5::HDF5 INTERFACE IMPORTED)
            set_target_properties(HDF5::HDF5 PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${HDF5_C_INCLUDE_DIRS}"
                INTERFACE_LINK_LIBRARIES "${HDF5_C_LIBRARIES}"
                INTERFACE_COMPILE_DEFINITIONS "${HDF5_C_DEFINITIONS}")
        endif()
        set(PYCUAMPCOR_WITH_HDF5 ON)
        message(STATUS "Reading images from HDF5 datasets with HDF5 ${HDF5_VERSION}")
    else()
        message(STATUS "HDF5 or zlib is not found; images can't be read from HDF5 datasets")
    endif()
endif()

# python and pybind11 requirement
# prefer the python in the active (e.g., conda) environment over newer system versions
set(Python_FIND_UNVERSIONED_NAMES FIRST)
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON) # Use new FindPython if available
find_package(pybind11 CONFIG REQUIRED)

# source lists
set(PYCUAMPCOR_CXX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cxx/pycuampcor)
include(${PYCUAMPCOR_CXX_DIR}/common/Sources.cmake)
include(${PYCUAMPCOR_CXX_DIR}/cpu/Sources.cmake)
include(${PYCUAMPCOR_CXX_DIR}/cuda/Sources.cmake)
if(PYCUAMPCOR_WITH_HDF5)
    list(APPEND PYCUAMPCOR_COMMON_SOURCES Hdf5SlcImage.cpp)
endif()
list(TRANSFORM PYCUAMPCOR_COMMON_SOURCES PREPEND ${PYCUAMPCOR_CXX_DIR}/common/)
list(TRANSFORM PYCUAMPCOR_CPU_SOURCES PREPEND ${PYCUAMPCOR_CXX_DIR}/cpu/)
list(TRANSFORM PYCUAMPCOR_CUDA_SOURCES PREPEND ${PYCUAMPCOR_CXX_DIR}/cuda/)

# add a static library pycuampcor_<backend> with the backend sources and
# the shared code in common/ (compiled once for each backend), and
# a python extension module pycuampcor._<backend> exposing class <pyclass>
function(pycuampcor_add_backend backend pyclass)
    string(TOUPPER ${backend} BACKEND)
    set(lib pycuampcor_${backend})
    add_library(${lib} STATIC
        ${PYCUAMPCOR_COMMON_SOURCES}
        ${PYCUAMPCOR_${BACKEND}_SOURCES}
        )
    set_target_properties(${lib} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        CUDA_VISIBILITY_PRESET hidden
        )
    target_include_directories(${lib} PUBLIC
        ${PYCUAMPCOR_CXX_DIR}/common
        ${PYCUAMPCOR_CXX_DIR}/${backend}
        )
    target_compile_definitions(${lib} PUBLIC
        PYCUAMPCOR_BACKEND=${backend}
        PYCUAMPCOR_BACKEND_${BACKEND}
        )
    if(PYCUAMPCOR_DOUBLE)
        target_compile_definitions(${lib} PUBLIC CUAMPCOR_DOUBLE)
    endif()
    if(PYCUAMPCOR_WITH_HDF5)
        target_compile_definitions(${lib} PUBLIC PYCUAMPCOR_WITH_HDF5)
        target_link_libraries(${lib} PUBLIC HDF5::HDF5 ZLIB::ZLIB)
    endif()

    set(target _${backend})
    pybind11_add_module(${target}
        ${CMAKE_CURRENT_SOURCE_DIR}/python/extensions/pycuampcor.cpp
        )
    target_compile_definitions(${target} PRIVATE
        PYCUAMPCOR_MODULE=${target}
        PYCUAMPCOR_CLASS=${pyclass}
        )
    target_link_libraries(${target} PRIVATE ${lib})
    list(APPEND PYCUAMPCOR_TARGETS ${target})
    set(PYCUAMPCOR_TARGETS ${PYCUAMPCOR_TARGETS} PARENT_SCOPE)
    list(APPEND PYCUAMPCOR_BACKENDS ${backend})
    set(PYCUAMPCOR_BACKENDS ${PYCUAMPCOR_BACKENDS} PARENT_SCOPE)
endfunction()

pycuampcor_add_backend(cpu PyCPUAmpcor)
target_link_libraries(pycuampcor_cpu PUBLIC ${PYCUAMPCOR_FFTW_TARGET})
if(OpenMP_CXX_FOUND)
    target_link_libraries(pycuampcor_cpu PUBLIC OpenMP::OpenMP_CXX)
endif()

if(PYCUAMPCOR_WITH_CUDA)
    pycuampcor_add_backend(cuda PyCuAmpcor)
    target_link_libraries(pycuampcor_cuda PUBLIC CUDA::cudart CUDA::cufft)
endif()

# C++ unit tests (with GoogleTest)
if(PYCUAMPCOR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/cxx)
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

# version file
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/_version.py "__version__ = \"${PROJECT_VERSION}\"\n")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_version.py
        DESTINATION ${PYCUAMPCOR_INSTALL_DIR})
