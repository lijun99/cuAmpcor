#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuAmpcorController.h"
#include "cuAmpcorParameter.h"
#include "SlcImage.h"
#ifdef PYCUAMPCOR_BACKEND_CUDA
#include "cudaUtil.h"
#endif

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace pycuampcor::PYCUAMPCOR_BACKEND;
using cls = cuAmpcorController;

// setters of the parameters by their python names, for the keyword constructor
using Setters = std::map<std::string, std::function<void(cls&, pybind11::handle)>>;

// convert a python value to a parameter value; file names may be path-like objects
template <typename T>
T toParam(pybind11::handle value)
{
    return value.cast<T>();
}

template <>
std::string toParam<std::string>(pybind11::handle value)
{
    return pybind11::module_::import("os").attr("fspath")(value).cast<std::string>();
}

// the python module with the options (enums) of the parameters
pybind11::module_ options()
{
    return pybind11::module_::import("pycuampcor.options");
}

// define a getter/setter for a controller parameter, converting the values with
// toPython (parameter -> python) and fromPython (python -> parameter), and register the setter
// a C++ value or a python object, as a python object
template <typename V>
pybind11::object asObject(V &&value)
{
    if constexpr (std::is_base_of_v<pybind11::handle, std::decay_t<V>>)
        return pybind11::reinterpret_borrow<pybind11::object>(value);
    else
        return pybind11::cast(std::forward<V>(value));
}

// the default value of a parameter, for the docstrings
std::string defaultText(pybind11::handle value)
{
    // an option as Option.MEMBER
    if (pybind11::isinstance(value, options().attr("_Option")))
        return value.attr("__class__").attr("__name__").cast<std::string>() + "."
               + value.attr("name").cast<std::string>();
    return pybind11::repr(value);
}

template <typename T, typename ToPython, typename FromPython>
void defParam(pybind11::class_<cls> &c, Setters &setters, const char *name,
              T cuAmpcorParameter::*member, const char *doc,
              ToPython toPython, FromPython fromPython)
{
    static const cuAmpcorParameter defaults;
    auto setter = [member, fromPython, name = std::string(name)](cls& self, pybind11::handle value) {
        auto invalid = [&]() {
            return "invalid value " + pybind11::repr(value).cast<std::string>()
                   + " for parameter '" + name + "'";
        };
        try {
            self.param->*member = fromPython(value);
        } catch (const pybind11::cast_error &) {
            throw pybind11::type_error(invalid());
        } catch (const pybind11::error_already_set &e) {
            // e.g., os.fspath of a number, or an invalid option name
            const std::string reason = pybind11::str(e.value());
            if (e.matches(PyExc_TypeError))
                throw pybind11::type_error(invalid() + ": " + reason);
            if (e.matches(PyExc_ValueError))
                throw pybind11::value_error(invalid() + ": " + reason);
            throw;
        }
    };
    c.def_property(name, [member, toPython](const cls& self) {
        return toPython(self.param->*member);
    }, [setter](cls& self, pybind11::object value) {
        setter(self, value);
    }, (std::string(doc) + "\n\nDefault: " + defaultText(asObject(toPython(defaults.*member)))).c_str());
    setters[name] = setter;
}

// a parameter of type T
template <typename T>
void defParam(pybind11::class_<cls> &c, Setters &setters, const char *name,
              T cuAmpcorParameter::*member, const char *doc)
{
    defParam(c, setters, name, member, doc, [](const T &value) { return value; }, toParam<T>);
}

// an option (an int parameter), read as a member of the enum pycuampcor.options.<option>,
// set by the member, the integer value, or the name
void defOption(pybind11::class_<cls> &c, Setters &setters, const char *name,
               int cuAmpcorParameter::*member, const char *option, const char *doc)
{
    defParam(c, setters, name, member, doc,
        [option](int value) { return options().attr(option)(value); },
        [option](pybind11::handle value) {
            return options().attr(option).attr("from_value")(value).cast<int>();
        });
}

// a boolean flag (an int parameter, 0 or 1), set by a bool or an integer
void defFlag(pybind11::class_<cls> &c, Setters &setters, const char *name,
             int cuAmpcorParameter::*member, const char *doc)
{
    defParam(c, setters, name, member, doc,
        [](int value) { return value != 0; },
        [](pybind11::handle value) {
            return static_cast<int>(options().attr("_to_flag")(value).cast<bool>());
        });
}

void setConstantGrossOffset(cls& self, const int goDown, const int goAcross)
{
    self.param->setStartPixels(
            self.param->referenceStartPixelDown0,
            self.param->referenceStartPixelAcross0,
            goDown, goAcross);
}

void setVaryingGrossOffset(cls& self, const std::vector<int> &vD, const std::vector<int> &vA)
{
    if ((int)vD.size() != self.param->numberWindows || (int)vA.size() != self.param->numberWindows)
        throw std::invalid_argument("The size of gross offsets does not match the number of windows");
    self.param->setStartPixels(
            self.param->referenceStartPixelDown0,
            self.param->referenceStartPixelAcross0,
            vD.data(), vA.data());
}

// set a constant (two numbers) or varying (two sequences, or a number and a sequence) gross offset
void setGrossOffset(cls& self, pybind11::handle goDown, pybind11::handle goAcross)
{
    const bool varyingDown = pybind11::isinstance<pybind11::sequence>(goDown);
    const bool varyingAcross = pybind11::isinstance<pybind11::sequence>(goAcross);
    if (!varyingDown && !varyingAcross) {
        int down, across;
        try {
            down = goDown.cast<int>();
            across = goAcross.cast<int>();
        } catch (const pybind11::cast_error &) {
            throw pybind11::type_error("gross offsets must be integers or sequences of integers");
        }
        setConstantGrossOffset(self, down, across);
        return;
    }
    const int n = self.param->numberWindows;
    auto toVector = [n](pybind11::handle v, bool varying) {
        return varying ? v.cast<std::vector<int>>() : std::vector<int>(n, v.cast<int>());
    };
    std::vector<int> vD, vA;
    try {
        vD = toVector(goDown, varyingDown);
        vA = toVector(goAcross, varyingAcross);
    } catch (const pybind11::cast_error &) {
        throw pybind11::type_error("gross offsets must be integers or sequences of integers");
    }
    setVaryingGrossOffset(self, vD, vA);
}

} // namespace

PYBIND11_MODULE(PYCUAMPCOR_MODULE, m)
{
    m.doc() = "Python module controller for underlying ampcor code";

    using str = std::string;

    // whether images can be read from HDF5 datasets (HDF5:<file>:<dataset>)
    m.attr("has_hdf5") = SlcImage::hasHdf5();

    pybind11::class_<cls> c(m, PYBIND11_TOSTRING(PYCUAMPCOR_CLASS),
        "Amplitude cross-correlation (ampcor) of a reference and a secondary image\n\n"
        "The offsets are estimated in a grid of numberWindowDown x numberWindowAcross\n"
        "windows; the parameters are the properties, set one by one (then setupParams and\n"
        "the gross offsets) or given to the constructor as keyword arguments.");
    Setters setters;

    c.def(pybind11::init<>(),
        "Create an ampcor object with the default parameters; set the parameters,\n"
        "then call setupParams() and setConstantGrossOffset() or setVaryingGrossOffset()\n"
        "before runAmpcor().");

    // define a getter/setter for a controller parameter, with its docstring
#define DEF_PARAM_RENAME(T, pyname, cppname, doc) \
    defParam<T>(c, setters, #pyname, &cuAmpcorParameter::cppname, doc)

    // same as above, for even more trivial cases where pyname == cppname
#define DEF_PARAM(T, name, doc) DEF_PARAM_RENAME(T, name, name, doc)

    // an option, as a member of the enum pycuampcor.options.<option>
#define DEF_OPTION(name, option, doc) \
    defOption(c, setters, #name, &cuAmpcorParameter::name, #option, doc)

    // a boolean flag
#define DEF_FLAG(name, doc) defFlag(c, setters, #name, &cuAmpcorParameter::name, doc)

    // processing
    DEF_OPTION(algorithm, Algorithm,
        "Cross-correlation domain (Algorithm): 'frequency' (FFT) or 'time'.");
    DEF_PARAM(int, deviceID,
        "CUDA device to use (GPU only); CUDA_VISIBLE_DEVICES also selects the devices.");
    DEF_PARAM(int, nStreams,
        "Number of CUDA streams (GPU only), each processing a chunk, to overlap the kernels\n"
        "with the data transfers; 2 is usually enough, more streams use more memory.");
    DEF_PARAM(int, nThreads,
        "Number of threads (CPU only), each processing a chunk; 0 for the OpenMP default\n"
        "(e.g., OMP_NUM_THREADS).");
    DEF_OPTION(derampMethod, DerampMethod,
        "Phase ramp removal before the antialiasing oversampling (DerampMethod):\n"
        "'magnitude' (take the amplitudes, e.g., for TOPS), 'complex' (remove a linear\n"
        "phase ramp) or 'none'.");
    DEF_OPTION(derampAxis, DerampAxis,
        "Axis of the phase ramp to remove (DerampAxis): 'down' (azimuth), 'across' (range)\n"
        "or 'both'.");
    DEF_OPTION(workflow, Workflow,
        "Cross-correlation workflow (Workflow): 'two_pass' (a first pass without the\n"
        "antialiasing oversampling) or 'one_pass'.");

    // images
    DEF_PARAM(str, referenceImageName,
        "Reference (template) image: a raw binary file, or HDF5:<file>:<dataset>.");
    DEF_PARAM(str, referenceImageReader,
        "Reader of the reference image: 'auto' (by the name), 'raw' or 'hdf5'.");
    DEF_PARAM(int, referenceImageHeight, "Height (lines) of the reference image.");
    DEF_PARAM(int, referenceImageWidth, "Width (columns) of the reference image.");
    DEF_OPTION(referenceImageDataType, DataType,
        "Pixel type of the reference image (DataType): 'complex64' or 'float32'.");
    DEF_PARAM(str, secondaryImageName,
        "Secondary (search) image: a raw binary file, or HDF5:<file>:<dataset>.");
    DEF_PARAM(str, secondaryImageReader,
        "Reader of the secondary image: 'auto' (by the name), 'raw' or 'hdf5'.");
    DEF_PARAM(int, secondaryImageHeight, "Height (lines) of the secondary image.");
    DEF_PARAM(int, secondaryImageWidth, "Width (columns) of the secondary image.");
    DEF_OPTION(secondaryImageDataType, DataType,
        "Pixel type of the secondary image (DataType): 'complex64' or 'float32'.");

    // windows
    DEF_PARAM(int, numberWindowDown, "Number of windows down (along the lines, azimuth).");
    DEF_PARAM(int, numberWindowAcross, "Number of windows across (along the columns, range).");
    DEF_PARAM_RENAME(int, windowSizeHeight, windowSizeHeightRaw,
        "Height of the reference (template) windows, in pixels.");
    DEF_PARAM_RENAME(int, windowSizeWidth, windowSizeWidthRaw,
        "Width of the reference (template) windows, in pixels.");
    DEF_PARAM_RENAME(int, halfSearchRangeDown, halfSearchRangeDownRaw,
        "Half of the search range down, in pixels: the secondary windows are larger by\n"
        "twice this range.");
    DEF_PARAM_RENAME(int, halfSearchRangeAcross, halfSearchRangeAcrossRaw,
        "Half of the search range across, in pixels: the secondary windows are larger by\n"
        "twice this range.");
    DEF_PARAM_RENAME(int, skipSampleDown, skipSampleDownRaw,
        "Distance between neighboring windows down, in pixels.");
    DEF_PARAM_RENAME(int, skipSampleAcross, skipSampleAcrossRaw,
        "Distance between neighboring windows across, in pixels.");
    DEF_PARAM_RENAME(int, referenceStartPixelDownStatic, referenceStartPixelDown0,
        "First line of the first reference window.");
    DEF_PARAM_RENAME(int, referenceStartPixelAcrossStatic, referenceStartPixelAcross0,
        "First column of the first reference window.");
    DEF_PARAM(int, numberWindowDownInChunk,
        "Number of windows down in a chunk (processed together); 0 for automatic:\n"
        "(SM/4) on the GPU, 1 on the CPU.");
    DEF_PARAM(int, numberWindowAcrossInChunk,
        "Number of windows across in a chunk (processed together); 0 for automatic:\n"
        "8 on the GPU, 1 on the CPU.");

    // oversampling and correlation surface
    DEF_PARAM(int, rawDataOversamplingFactor,
        "Antialiasing oversampling factor of the windows (2 for SLCs).");
    DEF_PARAM_RENAME(int, corrSurfaceZoomInWindow, zoomWindowSize,
        "Size of the correlation surface (at the antialiasing-oversampled spacing) around\n"
        "the peak to oversample (16 as in ROI_PAC).");
    DEF_PARAM(int, corrSurfaceOverSamplingFactor,
        "Oversampling factor of the correlation surface.");
    DEF_OPTION(corrSurfaceOverSamplingMethod, OversamplingMethod,
        "Oversampling method of the correlation surface (OversamplingMethod): 'fft' or 'sinc'.");
    DEF_PARAM(int, corrStatWindowSize,
        "Size of the correlation surface around the peak used for the SNR and the\n"
        "covariance.");

    // outputs
    DEF_PARAM(str, offsetImageName,
        "Output dense offsets: float32 (down, across) pairs, numberWindowDown x\n"
        "numberWindowAcross.");
    DEF_PARAM(str, grossOffsetImageName,
        "Output gross offsets: float32 (down, across) pairs.");
    DEF_FLAG(mergeGrossOffset,
        "Whether to add the gross offsets to the dense offsets.");
    DEF_PARAM(str, snrImageName, "Output signal-to-noise ratio of the correlation peak: float32.");
    DEF_PARAM(str, covImageName,
        "Output covariance of the offsets: float32 (down variance, across variance,\n"
        "covariance) triples.");
    DEF_PARAM(str, peakValueImageName,
        "Output normalized correlation peak value: float32.");
    // alias used by isce3 (v1)
    DEF_PARAM_RENAME(str, corrImageName, peakValueImageName,
        "Same as peakValueImageName (the name used by isce3).");

    // image i/o
    DEF_FLAG(useMmap, "Unused (raw images are always read with a memory map); kept for compatibility.");
    DEF_PARAM_RENAME(int, mmapSize, mmapSizeInGB,
        "Size in GB of the memory map (raw images) or of the cache of decoded chunks\n"
        "(HDF5 images); enlarged if a chunk needs more.");

    c.def(pybind11::init([setters](pybind11::kwargs kwargs) {
            auto self = std::make_unique<cls>();
            pybind11::object goDown = pybind11::int_(0), goAcross = pybind11::int_(0);
            for (const auto &[key, value] : kwargs) {
                const auto name = key.cast<std::string>();
                if (name == "grossOffsetDown")
                    goDown = pybind11::reinterpret_borrow<pybind11::object>(value);
                else if (name == "grossOffsetAcross")
                    goAcross = pybind11::reinterpret_borrow<pybind11::object>(value);
                else if (auto setter = setters.find(name); setter != setters.end())
                    setter->second(*self, value);
                else
                    throw pybind11::type_error("unexpected parameter '" + name + "'");
            }
            self->param->setupParameters();
            setGrossOffset(*self, goDown, goAcross);
            return self;
        }),
        "Create an ampcor object from parameters given as keyword arguments, named as the\n"
        "properties (e.g., windowSizeHeight=64, referenceImageName='ref.slc'), the others\n"
        "keeping their default values, and set it up to run (setupParams and gross offsets).\n"
        "The gross offsets grossOffsetDown and grossOffsetAcross (default 0) are constant\n"
        "(numbers) or varying (sequences of numberWindowDown*numberWindowAcross numbers).");

    c
        .def("runAmpcor", &cls::runAmpcor,
        "Run ampcor and write the outputs; the object must be set up (setupParams and\n"
        "the gross offsets, or the keyword constructor).")
        .def_static("runAmpcorLayers", [](const std::vector<cls*> &layers) {
            cls::runAmpcorLayers(layers);
        },
        "Run several layers (e.g., different window sizes) with the same images and\n"
        "numbers of windows (and windows per chunk), loading each chunk once for all layers.\n"
        "Each layer is configured (setupParams, gross offsets) as for runAmpcor;\n"
        "the device and the number of streams/threads are taken from the first layer.",
        pybind11::arg("layers"))

        .def("isDoublePrecision", &cls::isDoublePrecision,
        "Whether the computations and the outputs are in double precision (float64).")

        .def("checkPixelInImageRange", [](const cls& self) {
            self.param->checkPixelInImageRange();
        },
        "Warn about windows outside the images (in debug builds only); windows may extend\n"
        "beyond the images, which are padded with zeros.")

        .def("setupParams", [](cls& self) {
            self.param->setupParameters();
        },
        "Set up the processing from the parameters, after they are set; the gross offsets\n"
        "are then set with setConstantGrossOffset or setVaryingGrossOffset.")

        .def_property_readonly("windowsDownInChunk", [](const cls& self) {
            return self.param->windowsDownInChunk;
        },
        "Number of windows down in a chunk used, set by setupParams: numberWindowDownInChunk,\n"
        "or the automatic value if 0 (0 before setupParams).")
        .def_property_readonly("windowsAcrossInChunk", [](const cls& self) {
            return self.param->windowsAcrossInChunk;
        },
        "Number of windows across in a chunk used, set by setupParams:\n"
        "numberWindowAcrossInChunk, or the automatic value if 0 (0 before setupParams).")

        .def("setConstantGrossOffset", &setConstantGrossOffset,
        "Set a constant gross offset (in pixels) for all windows, after setupParams.",
        pybind11::arg("grossOffsetDown"), pybind11::arg("grossOffsetAcross"))
        .def("setVaryingGrossOffset", &setVaryingGrossOffset,
        "Set the gross offsets (in pixels) of each window, after setupParams, as two\n"
        "sequences of numberWindowDown*numberWindowAcross integers (row-major).",
        pybind11::arg("grossOffsetDown"), pybind11::arg("grossOffsetAcross"))

#ifdef PYCUAMPCOR_BACKEND_CUDA
        .def_static("device_init", &gpuDeviceInit,
        "Select the given cuda device for the calling thread; returns its id.",
        pybind11::arg("device") = 0)

        .def_static("get_sm_count", &getSMCount,
        "Returns the number of SMs (streaming multiprocessors) on the given device.",
        pybind11::arg("device") = 0)

        .def_static("device_list", [] {
            pybind11::list devices;
            for (const auto &d : gpuDeviceList())
                devices.append(pybind11::dict(
                    pybind11::arg("id") = d.id, pybind11::arg("name") = d.name,
                    pybind11::arg("sm_count") = d.smCount,
                    pybind11::arg("compute_capability") = pybind11::make_tuple(d.major, d.minor),
                    pybind11::arg("total_memory") = d.totalMemory));
            return devices;
        },
        "List the available cuda devices, as dicts with id, name, sm_count,\n"
        "compute_capability (major, minor) and total_memory (bytes)")

#endif
    ;
}
