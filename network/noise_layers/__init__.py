import random
from typing import List

def get_random_float(float_range: List[float]) -> float:
	return random.random() * (float_range[1] - float_range[0]) + float_range[0]


# def get_random_int(int_range: [int]):
def get_random_int(int_range: List[int]) -> int:
	return random.randint(int_range[0], int_range[1])



from .identity import Identity
from .crop import FaceCrop, FaceCropout, Dropout, FaceErase, FaceEraseout
from .salt_pepper import SaltPepper
from .jpeg import JpegTest
from .resize import Resize
from .kornia_noises import GaussianBlur, GaussianNoise, MedianBlur, Brightness, Contrast, Saturation, Hue, Rotation, Affine
############################################
from .simswap.test_one_image import SimSwap
from .ganimation.main import GANimation
from .stargan.main import StarGAN

