"""Stable project primitive-library construction."""

from kartezio.primitives.matrix import default_matrix_lib

from . import image_ops, morphology  # noqa: F401: register classes on import


def build_illumination_library():
    """Build the pilot-test matrix library without mutating global state."""
    library = default_matrix_lib()
    library.add_class(image_ops.OtsuThreshold)
    library.add_class(morphology.ClearBorder)
    return library
