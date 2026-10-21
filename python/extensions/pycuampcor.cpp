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
template <typename T, typename ToPython, typename FromPython>
void defParam(pybind11::class_<cls> &c, Setters &setters, const char *name,
              T cuAmpcorParameter::*member, ToPython toPython, FromPython fromPython)
{
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
    });
    setters[name] = setter;
}

// a parameter of type T
template <typename T>
void defParam(pybind11::class_<cls> &c, Setters &setters, const char *name,
              T cuAmpcorParameter::*member)
{
    defParam(c, setters, name, member, [](const T &value) { return value; }, toParam<T>);
}

// an option (an int parameter), read as a member of the enum pycuampcor.options.<option>,
// set by the member, the integer value, or the name
void defOption(pybind11::class_<cls> &c, Setters &setters, const char *name,
               int cuAmpcorParameter::*member, const char *option)
{
    defParam(c, setters, name, member,
        [option](int value) { return options().attr(option)(value); },
        [option](pybind11::handle value) {
            return options().attr(option).attr("from_value")(value).cast<int>();
        });
}

// a boolean flag (an int parameter, 0 or 1), set by a bool or an integer
void defFlag(pybind11::class_<cls> &c, Setters &setters, const char *name,
             int cuAmpcorParameter::*member)
{
    defParam(c, setters, name, member,
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

    pybind11::class_<cls> c(m, PYBIND11_TOSTRING(PYCUAMPCOR_CLASS));
    Setters setters;

    c.def(pybind11::init<>(),
        "Create an ampcor object with the default parameters; set the parameters,\n"
        "then call setupParams() and setConstantGrossOffset() or setVaryingGrossOffset()\n"
        "before runAmpcor().");

    // define a getter/setter for a controller parameter
#define DEF_PARAM_RENAME(T, pyname, cppname) \
    defParam<T>(c, setters, #pyname, &cuAmpcorParameter::cppname)

    // same as above, for even more trivial cases where pyname == cppname
#define DEF_PARAM(T, name) DEF_PARAM_RENAME(T, name, name)

    // an option, as a member of the enum pycuampcor.options.<option>
#define DEF_OPTION(name, option) \
    defOption(c, setters, #name, &cuAmpcorParameter::name, #option)

    // a boolean flag
#define DEF_FLAG(name) defFlag(c, setters, #name, &cuAmpcorParameter::name)

    DEF_OPTION(algorithm, Algorithm);
    DEF_PARAM(int, deviceID);
    DEF_PARAM(int, nStreams);
    DEF_PARAM(int, nThreads);
    DEF_OPTION(derampMethod, DerampMethod);
    DEF_OPTION(derampAxis, DerampAxis);
    DEF_OPTION(workflow, Workflow);

    DEF_PARAM(str, referenceImageName);
    DEF_PARAM(str, referenceImageReader);
    DEF_PARAM(int, referenceImageHeight);
    DEF_PARAM(int, referenceImageWidth);
    DEF_OPTION(referenceImageDataType, DataType);
    DEF_PARAM(str, secondaryImageName);
    DEF_PARAM(str, secondaryImageReader);
    DEF_PARAM(int, secondaryImageHeight);
    DEF_PARAM(int, secondaryImageWidth);
    DEF_OPTION(secondaryImageDataType, DataType);

    DEF_PARAM(int, numberWindowDown);
    DEF_PARAM(int, numberWindowAcross);

    DEF_PARAM_RENAME(int, windowSizeHeight, windowSizeHeightRaw);
    DEF_PARAM_RENAME(int, windowSizeWidth,  windowSizeWidthRaw);

    DEF_PARAM(str, offsetImageName);
    DEF_PARAM(str, grossOffsetImageName);
    DEF_FLAG(mergeGrossOffset);
    DEF_PARAM(str, snrImageName);
    DEF_PARAM(str, covImageName);
    DEF_PARAM(str, peakValueImageName);
    // alias used by isce3 (v1)
    DEF_PARAM_RENAME(str, corrImageName, peakValueImageName);

    DEF_PARAM(int, rawDataOversamplingFactor);
    DEF_PARAM(int, corrStatWindowSize);

    DEF_PARAM(int, numberWindowDownInChunk);
    DEF_PARAM(int, numberWindowAcrossInChunk);

    DEF_FLAG(useMmap);

    DEF_PARAM_RENAME(int, halfSearchRangeAcross, halfSearchRangeAcrossRaw);
    DEF_PARAM_RENAME(int, halfSearchRangeDown,   halfSearchRangeDownRaw);

    DEF_PARAM_RENAME(int, referenceStartPixelAcrossStatic, referenceStartPixelAcross0);
    DEF_PARAM_RENAME(int, referenceStartPixelDownStatic,   referenceStartPixelDown0);

    DEF_OPTION(corrSurfaceOverSamplingMethod, OversamplingMethod);
    DEF_PARAM(int, corrSurfaceOverSamplingFactor);

    DEF_PARAM_RENAME(int, mmapSize, mmapSizeInGB);

    DEF_PARAM_RENAME(int, skipSampleDown,   skipSampleDownRaw);
    DEF_PARAM_RENAME(int, skipSampleAcross, skipSampleAcrossRaw);
    DEF_PARAM_RENAME(int, corrSurfaceZoomInWindow, zoomWindowSize);

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
        // define a trivial binding for a controller method
#define DEF_METHOD(name) def(#name, &cls::name)

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

        .def("setConstantGrossOffset", &setConstantGrossOffset)
        .def("setVaryingGrossOffset", &setVaryingGrossOffset)

#ifdef PYCUAMPCOR_BACKEND_CUDA
        .def_static("device_init", &gpuDeviceInit,
        "Init the given cuda device (default = 0)",
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
