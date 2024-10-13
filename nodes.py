import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys
import math

from typing import Tuple, List
import sys
import os
import torch

from sklearn.cluster import KMeans

from numpy import ndarray
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class ColorWheelNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_string": ("STRING", {"default": "#ead5c2,#9a99ac,#e9b68b"}),
                "size": ("INT", {"default": 512, "min": 128, "max": 2048}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "create_accurate_color_wheel"
    CATEGORY = "image/generation"

    def hex_to_rgb(self, hex_color):
        return tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    def create_accurate_color_wheel(self, color_string, size):
        colors = [color.strip() for color in color_string.split(',')]
        output = np.zeros((size, size, 3), dtype=np.float32)
        value_map = np.zeros((size, size), dtype=np.float32)

        center = size // 2
        radius = int(size * 0.4)

        for y in range(size):
            for x in range(size):
                dx = x - center
                dy = y - center
                distance = math.sqrt(dx**2 + dy**2)
                if distance <= radius:
                    angle = (math.atan2(dy, dx) + math.pi) / (2 * math.pi)
                    saturation = distance / radius
                    r, g, b = colorsys.hsv_to_rgb(angle, saturation, 1)
                    h, s, v = colorsys.rgb_to_hsv(r, g, b)

                    vector_x = s * math.cos(h * 2 * math.pi)
                    vector_y = s * math.sin(h * 2 * math.pi)

                    img_x = int((vector_x + 1) * center)
                    img_y = int((vector_y + 1) * center)

                    if 0 <= img_x < size and 0 <= img_y < size:
                        if v > value_map[img_y, img_x]:
                            output[img_y, img_x] = [r, g, b]
                            value_map[img_y, img_x] = v

        img = Image.fromarray((output * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)

        for color in colors:
            rgb = self.hex_to_rgb(color)
            h, s, v = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)

            angle = h * 2 * math.pi
            distance = s * radius
            x = center + int(distance * math.cos(angle))
            y = center + int(distance * math.sin(angle))

            circle_radius = int(size * 0.02)
            draw.ellipse((x - circle_radius, y - circle_radius,
                          x + circle_radius, y + circle_radius), fill=color, outline='white')

        # Add palette block below the wheel
        palette_height = int(size * 0.2)  # Reduced height
        palette_width = size
        palette_block = Image.new('RGB', (palette_width, palette_height), color='white')
        palette_draw = ImageDraw.Draw(palette_block)

        font_size = int(size * 0.025)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        block_size = int(palette_height * 0.5)  # Smaller block size
        block_spacing = int(palette_height * 0.25)
        for i, color in enumerate(colors):
            block_x = i * (block_size + block_spacing) + block_spacing
            block_y = int(palette_height * 0.25)
            palette_draw.rectangle((block_x, block_y, block_x + block_size, block_y + block_size), fill=color)

            label_x = block_x
            label_y = block_y + block_size + int(palette_height * 0.05)
            palette_draw.text((label_x, label_y), color, fill='black', font=font)

        combined_img = Image.new('RGB', (size, size + palette_height), color='white')
        combined_img.paste(img, (0, 0))
        combined_img.paste(palette_block, (0, size))

        np_image = np.array(combined_img).astype(np.float32) / 255.0
        torch_image = torch.from_numpy(np_image)[None,]

        return (torch_image,)

class BK_Img2Color:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
            "optional": {
                "num_colors": ("INT", {"default": 1, "min": 1, }),
                "get_complementary_color": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_off": "false",
                        "label_on": "true",
                    },
                ),
                "accuracy": (
                    "INT",
                    {
                        "default": 80,
                        "display": "slider",
                        "min": 1,
                        "max": 100,
                    },
                ),
                "exclude_colors": (
                    "STRING",
                    {
                        "default": "",
                    },
                ),
                "select_color": ("INT", {
                    "default": 1,
                    "min": 1,
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("COLORS", "SELECT_COLOR",)
    CATEGORY = "⭐️ Baikong/Color"
    FUNCTION = "main"
    # OUTPUT_NODE = True
    DESCRIPTION = "从输入图像中提取主要颜色，可指定颜色数量，支持排除特定颜色，并可选择生成互补色"

    def __init__(self):
        pass

    def main(self, input_image: torch.Tensor, num_colors: int = 5, accuracy: int = 80,
             get_complementary_color: bool = False, exclude_colors: str = "", select_color: int = 1) -> Tuple[str, str]:
        self.exclude = [color.strip().lower() for color in exclude_colors.strip().split(
            ",")] if exclude_colors.strip() else []
        self.num_iterations = int(512 * (accuracy / 100))

        original_colors = self.interrogate_colors(input_image, num_colors)
        rgb = self.ndarrays_to_rgb(original_colors)

        if get_complementary_color:
            rgb = self.rgb_to_complementary(rgb)

        hex_colors = [
            f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}" for color in rgb]
        out = self.join_and_exclude(hex_colors)

        # 处理 select_color
        color_list = out.split(", ")
        selected_color = color_list[-1] if select_color > len(
            color_list) else color_list[select_color - 1]

        # 指定的输出格式 {"ui": {"text": (value1, value2)}, "result":  (value1, value2)}
        return {"ui": {"text": (out, selected_color)}, "result": (out, selected_color)}

    def join_and_exclude(self, colors: List[str]) -> str:
        return ", ".join(
            [str(color)
             for color in colors if color.lower() not in self.exclude]
        )

    def rgb_to_complementary(
        self, colors: List[Tuple[int, int, int]]
    ) -> List[Tuple[int, int, int]]:
        return [(255 - color[0], 255 - color[1], 255 - color[2]) for color in colors]

    def ndarrays_to_rgb(self, colors: List[ndarray]) -> List[Tuple[int, int, int]]:
        return [(int(color[0]), int(color[1]), int(color[2])) for color in colors]

    def interrogate_colors(self, image: torch.Tensor, num_colors: int) -> List[ndarray]:
        pixels = image.view(-1, image.shape[-1]).numpy()
        kmeans = KMeans(n_clusters=num_colors, algorithm="lloyd",
                        max_iter=self.num_iterations, n_init=10)
        colors = kmeans.fit(pixels).cluster_centers_ * 255
        return colors

NODE_CLASS_MAPPINGS = {
    "AccurateColorWheelNode": ColorWheelNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AccurateColorWheelNode": "Color Wheel Generator"
}
