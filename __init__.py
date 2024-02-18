import torch

class VideoFrameCrop:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict):
        Tell the main program input parameters of nodes.

    Attributes
    ----------
    RETURN_TYPES (`tuple`):
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "images": ("IMAGE",),
                "target_width": ("INT", {
                    "default": 512,
                    "min": 0,  # Minimum value
                    "max": 4096,  # Maximum value
                    "step": 64,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
                "target_height": ("INT", {
                    "default": 768,
                    "min": 0,  # Minimum value
                    "max": 4096,  # Maximum value
                    "step": 64,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
                "padding": ("INT", {
                    "default": 50,
                    "min": 0,  # Minimum value
                    "max": 4096,  # Maximum value
                    "step": 64,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "crop_centered_with_bbox"

    # OUTPUT_NODE = False

    CATEGORY = "VideoFrameCrop"

    def crop_centered_with_bbox(self, images, target_width, target_height, padding):
        """
        根据最小包围盒在给定的宽度和高度内居中裁剪图像，并在周围添加padding

        Args:
            images: 需要裁剪的图像，假设为PyTorch张量格式
            target_width: 目标裁剪区域的宽度（不包括padding）
            target_height: 目标裁剪区域的高度（不包括padding）
            padding: 裁剪区域边界的padding大小

        Returns:
            PyTorch张量: 居中裁剪并添加了padding后的图像
        """
        # 找到所有非零像素的坐标
        non_zero_pixels = torch.nonzero(images[0, :, :, :].any(dim=-1), as_tuple=True)
        y_min, x_min = torch.min(non_zero_pixels[0]), torch.min(non_zero_pixels[1])
        y_max, x_max = torch.max(non_zero_pixels[0]) + 1, torch.max(non_zero_pixels[1]) + 1

        # 计算最小包围盒的中心点
        bbox_center_x = (x_min + x_max) // 2
        bbox_center_y = (y_min + y_max) // 2

        # 考虑padding后计算裁剪区域的起始点
        start_x = max(bbox_center_x - (target_width // 2) - padding, 0)
        start_y = max(bbox_center_y - (target_height // 2) - padding, 0)

        # 考虑padding后确保裁剪区域不超出图像边界
        end_x = start_x + target_width + (2 * padding)
        end_y = start_y + target_height + (2 * padding)
        if end_x > images.shape[2]:
            end_x = images.shape[2]
            start_x = max(end_x - target_width - (2 * padding), 0)
        if end_y > images.shape[1]:
            end_y = images.shape[1]
            start_y = max(end_y - target_height - (2 * padding), 0)

        # 裁剪图像
        cropped_img = images[:, start_y:end_y, start_x:end_x, :]

        return (cropped_img,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VideoFrameCrop": VideoFrameCrop
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameCrop": "VideoFrameCrop"
}
