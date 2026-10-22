"""Project-specific image primitives missing from the upstream library."""

import numpy as np
from skimage.filters import threshold_otsu

from kartezio.core.components import Primitive, register
from kartezio.types import DataType, Matrix1


@register(Primitive)
class OtsuThreshold(Primitive):
    """Return a binary uint8 mask using the image's Otsu threshold."""

    def __init__(self):
        super().__init__(Matrix1, DataType.MATRIX, 0)

    def call(self, x, args):
        image = np.asarray(x[0])
        if image.ndim != 2:
            raise ValueError("OtsuThreshold expects a 2-D image")
        if np.all(image == image.flat[0]):
            return np.zeros(image.shape, dtype=np.uint8)
        threshold = threshold_otsu(image)
        return (image > threshold).astype(np.uint8) * 255
