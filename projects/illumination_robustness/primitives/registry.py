"""Stable primitive-library construction for illumination experiments."""

from kartezio.core.components import Library
from kartezio.primitives.matrix import (
    Close,
    Dilate,
    Erode,
    Fill,
    GaussianBlur,
    Identity,
    InRange,
    Inverse,
    MedianBlur,
    Normalize,
    Open,
    RmSmallObjects,
    Sobel,
    Threshold,
)
from kartezio.types import DataType

from . import image_ops, morphology  # noqa: F401: register classes on import


def build_illumination_library():
    """Build the fixed ``illumination_core_v1`` matrix library.

    The library is deliberately constructed from an explicit allow-list rather
    than starting with Kartezio's broad default matrix library. This keeps the
    primitive set identical across pilot and final experiments.
    """
    library = Library(DataType.MATRIX)
    for primitive in (
        Identity, Inverse, Normalize, GaussianBlur, MedianBlur,
        image_ops.OtsuThreshold, Threshold, InRange, Sobel,
        Erode, Dilate, Open, Close, Fill, RmSmallObjects,
        morphology.ClearBorder,
    ):
        library.add_class(primitive)
    return library
