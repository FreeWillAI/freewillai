import torch
import numpy as np
from numpy import ndarray
from PIL import Image as PILImage
from torchvision import transforms



class TorchTransform:

    def resize(self, image: PILImage.Image) -> ndarray:
        resized_image = transforms.Resize(image)
        return  np.array(resized_image)
