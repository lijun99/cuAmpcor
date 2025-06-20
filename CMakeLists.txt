# -*- cmake -*-
# Example with conda:
#   cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} -DCMAKE_BUILD_TYPE=Release

# minimum cmake version to support CUDA and Python
cmake_minimum_required(VERSION 3.18)

# project name
project(pycuampcor LANGUAGES CUDA CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 11)

# CUDA requirement
find_package(CUDAToolkit COMPONENTS cudart cufft REQUIRED)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 52 60 70 75 80 86 90)
endif()

# python and pybind11 requirement
find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)
set(PYBIND11_PYTHON_VERSION ${Python_VERSION})
set(PYBIND11_FINDPYTHON ON) # Use new FindPython if available
find_package(pybind11 CONFIG REQUIRED)

# python module
pybind11_add_module(pycuampcor
    src/PyCuAmpcor.cpp
    src/SlcImage.cpp
    src/cuAmpcorController.cpp
    src/cuAmpcorParameter.cpp
    src/cuArrays.cpp
    src/cuArraysCopy.cu
    src/cuArraysPadding.cu
    src/cuCorrFrequency.cu
    src/cuCorrNormalization.cu
    src/cuCorrNormalizationSAT.cu
    src/cuCorrNormalizer.cpp
    src/cuCorrTimeDomain.cu
    src/cuDeramp.cu
    src/cuEstimateStats.cu
    src/cuOffset.cu
    src/cuOverSampler.cpp
    src/cuSincOverSampler.cu
    src/cudaError.cpp
    src/cudaUtil.cpp
    src/cuAmpcorProcessor.cpp
    src/cuAmpcorProcessorTwoPass.cpp
    src/cuAmpcorProcessorOnePass.cpp
    )

# set_property(TARGET pycuampcor PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(pycuampcor PRIVATE
    src
    )

target_link_libraries(pycuampcor PRIVATE
    CUDA::cufft
    )

# Condition to check if using skbuild
if (SKBUILD)
    # Install package
    install(TARGETS pycuampcor
            LIBRARY DESTINATION pycuampcor)
else()
    # Install package
    if(NOT DEFINED PYCUAMPCOR_INSTALL_DIR)
        set(PYCUAMPCOR_INSTALL_DIR ${Python_SITEARCH}/pycuampcor)
    endif()

    # Install compiled library
    install(TARGETS pycuampcor
            LIBRARY DESTINATION ${PYCUAMPCOR_INSTALL_DIR})

    # Install __init__.py
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/pycuampcor/__init__.py
            DESTINATION ${PYCUAMPCOR_INSTALL_DIR})
    message(STATUS "Installing pycuampcor to: ${PYCUAMPCOR_INSTALL_DIR}")
endif()







