# -*- cmake -*-
# Example with conda:
#   cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} -DCMAKE_BUILD_TYPE=Release

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

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75 80 86 90)
endif()

# python and pybind11 requirement
find_package(Python 3.6 COMPONENTS Interpreter Development REQUIRED)
set(PYBIND11_PYTHON_VERSION ${Python_VERSION})
set(PYBIND11_FINDPYTHON ON) # Use new FindPython if available
find_package(pybind11 CONFIG REQUIRED)

# gdal requirement
find_package(GDAL REQUIRED)

# python module
pybind11_add_module(PyCuAmpcor
    src/PyCuAmpcor.cpp
    src/GDALImage.cpp
    src/cuAmpcorChunk.cpp
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
    )

# set_property(TARGET PyCuAmpcor PROPERTY CUDA_SEPARABLE_COMPILATION ON)

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


if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

  install(TARGETS PyCuAmpcor DESTINATION .)

else()
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
endif()




