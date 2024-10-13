import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys
import math

class AccurateColorWheelNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_string": ("STRING", {"default": "#beb6bb,#c38064,#9a99ac,#e9b68b,#ead5c2"}),
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
                    
                    # Convert HSV to 2D vector
                    vector_x = s * math.cos(h * 2 * math.pi)
                    vector_y = s * math.sin(h * 2 * math.pi)
                    
                    # Map to image coordinates
                    img_x = int((vector_x + 1) * center)
                    img_y = int((vector_y + 1) * center)
                    
                    if 0 <= img_x < size and 0 <= img_y < size:
                        if v > value_map[img_y, img_x]:
                            output[img_y, img_x] = [r, g, b]
                            value_map[img_y, img_x] = v

        # Create PIL Image for drawing
        img = Image.fromarray((output * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)

        # Place color markers and labels
        for color in colors:
            rgb = self.hex_to_rgb(color)
            h, s, v = colorsys.rgb_to_hsv(rgb[0]/255, rgb[1]/255, rgb[2]/255)
            
            angle = h * 2 * math.pi
            distance = s * radius
            x = center + int(distance * math.cos(angle))
            y = center + int(distance * math.sin(angle))

            # Draw color circle
            circle_radius = int(size * 0.03)
            draw.ellipse((x - circle_radius, y - circle_radius,
                          x + circle_radius, y + circle_radius), fill=color, outline='white')

            # Add color label
            label_distance = radius + int(size * 0.08)
            label_x = center + int(label_distance * math.cos(angle))
            label_y = center + int(label_distance * math.sin(angle))
            draw.text((label_x, label_y), color, fill='black', anchor='mm', stroke_width=2, stroke_fill='white')

        # Convert to numpy array and then to PyTorch tensor
        np_image = np.array(img).astype(np.float32) / 255.0
        torch_image = torch.from_numpy(np_image)[None,]

        return (torch_image,)

NODE_CLASS_MAPPINGS = {
    "AccurateColorWheelNode": AccurateColorWheelNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AccurateColorWheelNode": "Accurate Color Wheel Generator"
}
