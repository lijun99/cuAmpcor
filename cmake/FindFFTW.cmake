# Find the FFTW3 library (single or double precision)
#
# Usage:
#   find_package(FFTW [REQUIRED] [COMPONENTS Float Double])
#
# Imported targets (for each found component):
#   FFTW::Float   libfftw3f
#   FFTW::Double  libfftw3
#
# Hints: FFTW_ROOT, CMAKE_PREFIX_PATH

include(FindPackageHandleStandardArgs)

find_path(FFTW_INCLUDE_DIR NAMES fftw3.h
    HINTS ${FFTW_ROOT} ENV FFTW_ROOT PATH_SUFFIXES include)

if(NOT FFTW_FIND_COMPONENTS)
    set(FFTW_FIND_COMPONENTS Float)
endif()

set(_fftw_Float_lib fftw3f)
set(_fftw_Double_lib fftw3)

foreach(comp ${FFTW_FIND_COMPONENTS})
    find_library(FFTW_${comp}_LIBRARY NAMES ${_fftw_${comp}_lib}
        HINTS ${FFTW_ROOT} ENV FFTW_ROOT PATH_SUFFIXES lib lib64)
    if(FFTW_${comp}_LIBRARY AND FFTW_INCLUDE_DIR)
        set(FFTW_${comp}_FOUND TRUE)
        if(NOT TARGET FFTW::${comp})
            add_library(FFTW::${comp} UNKNOWN IMPORTED)
            set_target_properties(FFTW::${comp} PROPERTIES
                IMPORTED_LOCATION ${FFTW_${comp}_LIBRARY}
                INTERFACE_INCLUDE_DIRECTORIES ${FFTW_INCLUDE_DIR})
        endif()
    endif()
    mark_as_advanced(FFTW_${comp}_LIBRARY)
endforeach()
mark_as_advanced(FFTW_INCLUDE_DIR)

find_package_handle_standard_args(FFTW
    REQUIRED_VARS FFTW_INCLUDE_DIR
    HANDLE_COMPONENTS)
