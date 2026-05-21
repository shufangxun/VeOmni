import numpy as np
from PIL import Image

from veomni.data.multimodal.image_utils import load_image


def test_load_image_accepts_pil_image():
    image = Image.new("RGBA", (4, 3), color=(255, 0, 0, 255))

    loaded = load_image(image)

    assert loaded.mode == "RGB"
    assert loaded.size == (4, 3)


def test_load_image_accepts_numpy_array():
    image = np.zeros((3, 4, 3), dtype=np.uint8)

    loaded = load_image(image)

    assert loaded.mode == "RGB"
    assert loaded.size == (4, 3)
