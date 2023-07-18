import numpy as np
import torch
from imwatermark import WatermarkEncoder


WATERMARK_MESSAGE = "synthica.ai"

class StableDiffusionXLWatermarker:
    def __init__(self):
        self.watermark = WATERMARK_MESSAGE
        self.encoder = WatermarkEncoder()

        self.encoder.set_watermark("bytes", self.watermark.encode('utf-8'))

    def apply_watermark(self, images: torch.FloatTensor):
        # can't encode images that are smaller than 256
        if images.shape[-1] < 256:
            return images

        images = (255 * (images / 2 + 0.5)).cpu().permute(0, 2, 3, 1).float().numpy()

        images = [self.encoder.encode(image, "dwtDct") for image in images]

        images = torch.from_numpy(np.array(images)).permute(0, 3, 1, 2)

        images = torch.clamp(2 * (images / 255 - 0.5), min=-1.0, max=1.0)
        return images