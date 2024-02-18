import torch
import torch.nn.functional as F


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
                    "default": 0,
                    "min": 0,  # Minimum value
                    "max": 4096,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number"  # Cosmetic only: display as "number" or "slider"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "crop_centered_with_bbox"

    # OUTPUT_NODE = False

    CATEGORY = "VideoFrameCrop"

    def debug(self, images):
        # 获取张量的形状
        shape = images.shape

        # 获取张量的数据类型
        dtype = images.dtype

        # 获取张量的设备（CPU或GPU）
        device = images.device

        # 如果张量有批量大小，获取批量大小
        batch_size = images.shape[0] if len(shape) == 4 else None

        # 获取通道数
        channel_num = images.shape[1] if len(shape) == 4 else images.shape[0]

        # 获取图像高度和宽度
        height = images.shape[-2]
        width = images.shape[-1]

        # 打印获取的信息
        print("Shape:", shape)
        print("Data type:", dtype)
        print("Device:", device)
        print("Batch size:", batch_size)
        print("Number of channels:", channel_num)
        print("Height:", height)
        print("Width:", width)

    def crop_centered_with_bbox(self, images, target_width, target_height, padding):
        # self.debug(images)

        if not isinstance(images, torch.Tensor):
            return (images,)

        # 假设images的形状为[1, 高度, 宽度, 3]
        non_zero_pixels = torch.nonzero(images[0, :, :, :].any(dim=-1), as_tuple=True)
        if len(non_zero_pixels[0]) == 0 or len(non_zero_pixels[1]) == 0:
            return (images,)

        y_min, x_min = torch.min(non_zero_pixels[0]), torch.min(non_zero_pixels[1])
        y_max, x_max = torch.max(non_zero_pixels[0]) + 1, torch.max(non_zero_pixels[1]) + 1

        cropped_img = images[:, y_min:y_max, x_min:x_max, :]

        # 等比例缩放图像
        original_height = y_max - y_min
        original_width = x_max - x_min
        height_ratio = (target_height - 2 * padding - 100) / original_height
        width_ratio = (target_width - 2 * padding) / original_width
        scale_ratio = min(height_ratio, width_ratio)

        scaled_height = int(original_height * scale_ratio)
        scaled_width = int(original_width * scale_ratio)

        # print(f'等比例缩放: w = {scaled_width}, h = {scaled_height}')

        scaled_img = F.interpolate(cropped_img.permute(0, 3, 1, 2).float(),
                                   size=(scaled_height, scaled_width),
                                   mode='bilinear',
                                   align_corners=False).permute(0, 2, 3, 1)

        # 创建新画布
        canvas = torch.zeros((1, target_height, target_width, 3), device=images.device, dtype=scaled_img.dtype)

        # print(f'新画布大小: w = {target_width}, h = {target_height}')

        # 计算缩放后图像在新画布上的中心对齐位置
        start_x = (target_width - scaled_width) // 2
        start_y = (target_height - scaled_height - 100) // 2  # 保持底部100像素间隔

        # 复制缩放后的图像到新画布
        canvas[:, start_y:start_y + scaled_height, start_x:start_x + scaled_width, :] = scaled_img

        return (canvas,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VideoFrameCrop": VideoFrameCrop
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoFrameCrop": "VideoFrameCrop"
}
