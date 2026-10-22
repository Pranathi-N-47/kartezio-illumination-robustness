"""Project-specific morphology primitives missing from Kartezio."""

import numpy as np
from skimage.segmentation import clear_border

from kartezio.core.components import Primitive, register
from kartezio.types import DataType, Matrix1


@register(Primitive)
class ClearBorder(Primitive):
    """Remove connected foreground components touching the image border."""

    def __init__(self):
        super().__init__(Matrix1, DataType.MATRIX, 0)

    def call(self, x, args):
        image = np.asarray(x[0])
        if image.ndim != 2:
            raise ValueError("ClearBorder expects a 2-D image")
        return clear_border(image > 0).astype(np.uint8) * 255
