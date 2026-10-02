#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuAmpcorController.h"
#include "cuAmpcorParameter.h"
#include "SlcImage.h"
#ifdef PYCUAMPCOR_BACKEND_CUDA
#include "cudaUtil.h"
#endif

PYBIND11_MODULE(PYCUAMPCOR_MODULE, m)
{
    m.doc() = "Python module controller for underlying ampcor code";

    using namespace pycuampcor::PYCUAMPCOR_BACKEND;
    using str = std::string;
    using cls = cuAmpcorController;

    // whether images can be read from HDF5 datasets (HDF5:<file>:<dataset>)
    m.attr("has_hdf5") = SlcImage::hasHdf5();

    pybind11::class_<cls>(m, PYBIND11_TOSTRING(PYCUAMPCOR_CLASS))
        .def(pybind11::init<>())

        // define a trivial binding for a controller method
#define DEF_METHOD(name) def(#name, &cls::name)

        // define a trivial getter/setter for a controller parameter
#define DEF_PARAM_RENAME(T, pyname, cppname) \
        def_property(#pyname, [](const cls& self) -> T { \
            return self.param->cppname; \
        }, [](cls& self, const T i) { \
            self.param->cppname = i; \
        })

        // same as above, for even more trivial cases where pyname == cppname
#define DEF_PARAM(T, name) DEF_PARAM_RENAME(T, name, name)

        .DEF_PARAM(int, algorithm)
        .DEF_PARAM(int, deviceID)
        .DEF_PARAM(int, nStreams)
        .DEF_PARAM(int, nThreads)
        .DEF_PARAM(int, derampMethod)
        .DEF_PARAM(int, derampAxis)
        .DEF_PARAM(int, workflow)

        .DEF_PARAM(str, referenceImageName)
        .DEF_PARAM(str, referenceImageReader)
        .DEF_PARAM(int, referenceImageHeight)
        .DEF_PARAM(int, referenceImageWidth)
        .DEF_PARAM(int, referenceImageDataType)
        .DEF_PARAM(str, secondaryImageName)
        .DEF_PARAM(str, secondaryImageReader)
        .DEF_PARAM(int, secondaryImageHeight)
        .DEF_PARAM(int, secondaryImageWidth)
        .DEF_PARAM(int, secondaryImageDataType)

        .DEF_PARAM(int, numberWindowDown)
        .DEF_PARAM(int, numberWindowAcross)

        .DEF_PARAM_RENAME(int, windowSizeHeight, windowSizeHeightRaw)
        .DEF_PARAM_RENAME(int, windowSizeWidth,  windowSizeWidthRaw)

        .DEF_PARAM(str, offsetImageName)
        .DEF_PARAM(str, grossOffsetImageName)
        .DEF_PARAM(int, mergeGrossOffset)
        .DEF_PARAM(str, snrImageName)
        .DEF_PARAM(str, covImageName)
        .DEF_PARAM(str, peakValueImageName)
        // alias used by isce3 (v1)
        .DEF_PARAM_RENAME(str, corrImageName, peakValueImageName)

        .DEF_PARAM(int, rawDataOversamplingFactor)
        .DEF_PARAM(int, corrStatWindowSize)

        .DEF_PARAM(int, numberWindowDownInChunk)
        .DEF_PARAM(int, numberWindowAcrossInChunk)

        .DEF_PARAM(int, useMmap)

        .DEF_PARAM_RENAME(int, halfSearchRangeAcross, halfSearchRangeAcrossRaw)
        .DEF_PARAM_RENAME(int, halfSearchRangeDown,   halfSearchRangeDownRaw)

        .DEF_PARAM_RENAME(int, referenceStartPixelAcrossStatic, referenceStartPixelAcross0)
        .DEF_PARAM_RENAME(int, referenceStartPixelDownStatic,   referenceStartPixelDown0)

        .DEF_PARAM(int, corrSurfaceOverSamplingMethod)
        .DEF_PARAM(int, corrSurfaceOverSamplingFactor)

        .DEF_PARAM_RENAME(int, mmapSize, mmapSizeInGB)

        .DEF_PARAM_RENAME(int, skipSampleDown,   skipSampleDownRaw)
        .DEF_PARAM_RENAME(int, skipSampleAcross, skipSampleAcrossRaw)
        .DEF_PARAM_RENAME(int, corrSurfaceZoomInWindow, zoomWindowSize)

        .DEF_METHOD(runAmpcor)
        .def_static("runAmpcorLayers", [](const std::vector<cls*> &layers) {
            cls::runAmpcorLayers(layers);
        },
        "Run several layers (e.g., different window sizes) with the same images and\n"
        "numbers of windows (and windows per chunk), loading each chunk once for all layers.\n"
        "Each layer is configured (setupParams, gross offsets) as for runAmpcor;\n"
        "the device and the number of streams/threads are taken from the first layer.",
        pybind11::arg("layers"))

        .DEF_METHOD(isDoublePrecision)

        .def("checkPixelInImageRange", [](const cls& self) {
            self.param->checkPixelInImageRange();
        })

        .def("setupParams", [](cls& self) {
            self.param->setupParameters();
        })

        .def("setConstantGrossOffset", [](cls& self, const int goDown,
                                                     const int goAcross) {
            self.param->setStartPixels(
                    self.param->referenceStartPixelDown0,
                    self.param->referenceStartPixelAcross0,
                    goDown, goAcross);
        })
        .def("setVaryingGrossOffset", [](cls& self, std::vector<int> vD,
                                                    std::vector<int> vA) {
            if ((int)vD.size() != self.param->numberWindows || (int)vA.size() != self.param->numberWindows)
                throw std::invalid_argument("The size of gross offsets does not match the number of windows");
            self.param->setStartPixels(
                    self.param->referenceStartPixelDown0,
                    self.param->referenceStartPixelAcross0,
                    vD.data(), vA.data());
        })

#ifdef PYCUAMPCOR_BACKEND_CUDA
        .def_static("device_init", [](int device = 0) {
            return gpuDeviceInit(device);
        },
        "Init the given cuda device (default = 0)")

        .def_static("get_sm_count", [](int device = 0) {
            return getSMCount(device);
        },
        "Returns the number of SMs (streaming multiprocessors) on the given device.")

        .def_static("device_list", &gpuDeviceList,
        "List all available cuda devices")

#endif
    ;
}
