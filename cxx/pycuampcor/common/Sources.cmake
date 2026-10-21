# the shared code (namespace pycuampcor), compiled once into pycuampcor_common
set(PYCUAMPCOR_COMMON_SOURCES
    SlcImage.cpp
    cuAmpcorLayers.cpp
    cuAmpcorParameter.cpp
    logging.cpp
    )
# the processors of the chunks: one algorithm, compiled for each package (cpu and cuda)
set(PYCUAMPCOR_PROCESSOR_SOURCES
    cuAmpcorProcessor.cpp
    cuAmpcorProcessorOnePass.cpp
    cuAmpcorProcessorTwoPass.cpp
    )
