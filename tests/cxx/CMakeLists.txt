# C++ unit tests for the processing kernels and helpers
#
# The test sources are backend-agnostic: they are compiled once for each backend
# (test_cpu, test_cuda), in the same way as the shared code in cxx/pycuampcor/common,
# so that both implementations are checked against the same references.

find_package(GTest REQUIRED)
include(GoogleTest)

set(PYCUAMPCOR_TEST_SOURCES
    test_arrays.cpp
    test_copy.cpp
    test_stats.cpp
    test_offset.cpp
    test_fft.cpp
    test_deramp.cpp
    test_sinc.cpp
    test_parameter.cpp
    test_slcimage.cpp
    )

foreach(backend ${PYCUAMPCOR_BACKENDS})
    set(target test_${backend})
    add_executable(${target} ${PYCUAMPCOR_TEST_SOURCES})
    target_link_libraries(${target} PRIVATE pycuampcor_${backend} GTest::gtest_main)
    gtest_discover_tests(${target} TEST_PREFIX ${backend}. DISCOVERY_MODE PRE_TEST)
endforeach()
