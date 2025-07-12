# Copyright (C) 2024 NVIDIA Corporation.  All rights reserved.
#
# This work is licensed under the LICENSE file
# located at the root directory.

from typing import List, Dict
import torch

import sys 
sys.path.append(".")
sys.path.append("..")

import numpy as np
from typing import List
from diffusers.utils.torch_utils import randn_tensor
import torch.nn.functional as F
from skimage import filters
from tqdm import tqdm

def attn_map_to_binary(attention_map, scaler=1.):
    
    attention_map_np = attention_map.detach().cpu().numpy()
    
    threshold_value = filters.threshold_otsu(attention_map_np) * scaler 
    binary_mask = (attention_map_np > threshold_value).astype(np.uint8)

    return binary_mask

def gaussian_smooth(input_tensor, kernel_size=3, sigma=1):
    """
    Function to apply Gaussian smoothing on each 2D slice of a 3D tensor.
    """

    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * 
                      np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )
    kernel = torch.Tensor(kernel / kernel.sum()).to(input_tensor.dtype).to(input_tensor.device)
    
    # Add batch and channel dimensions to the kernel
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    # Iterate over each 2D slice and apply convolution
    smoothed_slices = []
    for i in range(input_tensor.size(0)):
        slice_tensor = input_tensor[i, :, :]
        slice_tensor = F.conv2d(slice_tensor.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size // 2)[0, 0]
        smoothed_slices.append(slice_tensor)
    
    # Stack the smoothed slices to get the final tensor
    smoothed_tensor = torch.stack(smoothed_slices, dim=0)

    return smoothed_tensor

import cv2
import numpy as np
from typing import Tuple, List

def text_under_image(
    image: np.ndarray,
    text: str,
    text_color: Tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 2.5,
    add_h: float = 0.4,
    line_spacing: int = 30,
    max_width: int = None,
) -> np.ndarray:
    """
    Adds multiline text below an image, with automatic line wrapping.

    Args:
        image: Input image as a NumPy array.
        text: Text to render below the image.
        text_color: RGB color of the text. Default is black.
        font_scale: Scale of the text font. Default is 2.5.
        add_h: Extra vertical spacing between the image and the text block, as a ratio of image height.
        line_spacing: Pixel spacing between lines. Default is 30.
        max_width: Maximum line width in pixels. Defaults to the image width.

    Returns:
        A new image with the text rendered below the original image.
    """
    h, w, c = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    if max_width is None:
        max_width = w

    def split_text_into_lines(text: str, max_width: int) -> List[str]:
        lines = []
        current_line = ""
        for word in text.split():
            (line_width, _), _ = cv2.getTextSize(current_line + " " + word, font, font_scale, thickness)
            if line_width <= max_width:
                current_line += " " + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    lines = split_text_into_lines(text, max_width)

    (_, text_height), _ = cv2.getTextSize("Test", font, font_scale, thickness)
    total_text_height = len(lines) * (text_height + line_spacing)

    offset = int(h * add_h)
    new_h = h + offset + total_text_height
    img = np.ones((new_h, w, c), dtype=np.uint8) * 255
    img[:h] = image

    y = h + offset
    for line in lines:
        (line_width, line_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_x = (w - line_width) // 2  # center text horizontally
        cv2.putText(img, line, (text_x, y), font, font_scale, text_color, thickness)
        y += line_height + line_spacing

    return img


