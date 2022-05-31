# -*- cmake -*-
# minimum cmake version to support CUDA and Python
cmake_minimum_required(VERSION 3.18)

# project name
project(PyCuAmpcor LANGUAGES CUDA CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 11)

# CUDA requirement
find_package(CUDAToolkit COMPONENTS cudart cufft REQUIRED)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

# python and pybind11 requirement
find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)
set(PYBIND11_CPP_STANDARD -std=c++11)
set(PYBIND11_PYTHON_VERSION ${Python_VERSION})
set(PYBIND11_FINDPYTHON ON) # Use new FindPython if available
find_package(pybind11 CONFIG REQUIRED)

# gdal requirement
find_package(GDAL REQUIRED)

# python module
pybind11_add_module(PyCuAmpcor
    src/PyCuAmpcor.cpp
    src/GDALImage.cu
    src/cuAmpcorChunk.cu
    src/cuAmpcorController.cu
    src/cuAmpcorParameter.cu
    src/cuArrays.cu
    src/cuArraysCopy.cu
    src/cuArraysPadding.cu
    src/cuCorrFrequency.cu
    src/cuCorrNormalization.cu
    src/cuCorrNormalizationSAT.cu
    src/cuCorrNormalizer.cu
    src/cuCorrTimeDomain.cu
    src/cuDeramp.cu
    src/cuEstimateStats.cu
    src/cuOffset.cu
    src/cuOverSampler.cu
    src/cuSincOverSampler.cu
    )

target_include_directories(PyCuAmpcor PRIVATE
    src
    )

target_link_libraries(PyCuAmpcor PRIVATE
    CUDA::cufft
    GDAL::GDAL
    )

if(NOT DEFINED PYTHON_MODULE_DIR)
    set(PYTHON_MODULE_DIR packages CACHE PATH
        "Python module directory (relative to install prefix)")
endif()

# get the python package path
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "if True:
    from distutils import sysconfig as sc
    print(sc.get_python_lib(prefix='', plat_specific=True))"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE)

message(STATUS "installing PyCuAmpcor to \"${CMAKE_INSTALL_PREFIX}/${PYTHON_SITE_PACKAGES}\"")

install(TARGETS PyCuAmpcor
  LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
  )


