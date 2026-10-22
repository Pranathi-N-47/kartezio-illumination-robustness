import numpy as np

from projects.illumination_robustness.evolution.fitness import PilotTestQSegFitness
from projects.illumination_robustness.primitives import build_illumination_library
from projects.illumination_robustness.primitives.image_ops import OtsuThreshold
from projects.illumination_robustness.primitives.morphology import ClearBorder


def test_pilot_test_library_contains_project_primitives():
    library = build_illumination_library()
    names = [library.name_of(i) for i in range(library.size)]
    assert "OtsuThreshold" in names
    assert "ClearBorder" in names


def test_project_primitives_are_deterministic_and_shape_preserving():
    image = np.full((32, 32), 220, dtype=np.uint8)
    image[10:22, 10:22] = 20
    otsu = OtsuThreshold().call([image], [])
    assert otsu.shape == image.shape
    assert otsu.dtype == np.uint8
    np.testing.assert_array_equal(otsu, OtsuThreshold().call([image], []))

    border = np.zeros((32, 32), dtype=np.uint8)
    border[0:5, 0:5] = 255
    border[10:15, 10:15] = 255
    cleared = ClearBorder().call([border], [])
    assert cleared[2, 2] == 0
    assert cleared[12, 12] == 255


def test_pilot_test_fitness_prefers_perfect_prediction():
    truth = np.zeros((16, 16), dtype=np.uint8)
    truth[4:12, 4:12] = 255
    fitness = PilotTestQSegFitness()
    perfect = fitness.evaluate([[truth]], [[truth]])[0]
    empty = fitness.evaluate([[truth]], [[np.zeros_like(truth)]])[0]
    assert perfect == 0.0
    assert empty > perfect
