"""
Options of the ampcor parameters

Each option is an integer enum: its members equal the integer values used before
(e.g., Workflow.TWO_PASS == 0), so that an option parameter may be set by the member,
the integer value, or the name (case-insensitive, e.g., "two_pass" or "two-pass"),
and is read as the member. Some members have aliases, e.g., DerampAxis.AZIMUTH for
DerampAxis.DOWN.
"""

import enum
import operator

__all__ = ["Algorithm", "Workflow", "DerampMethod", "DerampAxis",
           "OversamplingMethod", "DataType"]


class _Option(enum.IntEnum):
    """An option, given by its member, integer value or name"""

    @classmethod
    def from_value(cls, value):
        """The member for a member, an integer value, or a name (case-insensitive)"""
        if isinstance(value, str):
            name = value.strip().upper().replace("-", "_")
            try:
                return cls[name]
            except KeyError:
                pass
            raise ValueError(f"invalid {cls.__name__} {value!r}; choose from "
                             + ", ".join(repr(n.lower()) for n in cls.__members__)) from None
        # integers (including members of an equivalent enum), not floats
        return cls(operator.index(value))


class Algorithm(_Option):
    """Cross-correlation domain"""
    FREQUENCY = 0  #: in the frequency domain, with FFTs (default)
    TIME = 1       #: in the time (spatial) domain
    FREQ = 0


class Workflow(_Option):
    """Cross-correlation workflow"""
    TWO_PASS = 0  #: a first pass without the antialiasing oversampling (default)
    ONE_PASS = 1  #: one pass with the antialiasing oversampling


class DerampMethod(_Option):
    """Removal of the phase ramp before the antialiasing oversampling"""
    MAGNITUDE = 0  #: take the amplitudes (e.g., for TOPS; default)
    COMPLEX = 1    #: remove a linear phase ramp, estimated from the average phase difference
    NONE = 2       #: no deramping
    MAG = 0
    LINEAR = 1
    SKIP = 2


class DerampAxis(_Option):
    """Axis (or axes) of the phase ramp to remove"""
    DOWN = 0    #: down (azimuth)
    ACROSS = 1  #: across (range)
    BOTH = 2    #: both (default)
    AZIMUTH = 0
    RANGE = 1


class OversamplingMethod(_Option):
    """Oversampling method of the correlation surface"""
    FFT = 0   #: FFT (default)
    SINC = 1  #: sinc interpolation


class DataType(_Option):
    """Pixel type of an image (the value is the number of floats in a pixel)"""
    FLOAT32 = 1    #: real (amplitude) images
    COMPLEX64 = 2  #: complex images (default)
    REAL = 1
    FLOAT = 1
    COMPLEX = 2


def _to_flag(value):
    """A boolean flag from a bool or an integer (not, e.g., a string or a float)"""
    if getattr(value, "dtype", None) is not None and value.dtype.kind == "b":
        return bool(value)
    return bool(operator.index(value))
