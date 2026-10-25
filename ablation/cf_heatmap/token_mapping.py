"""把展平的 LLM token 位置映射回图像空间网格（Qwen2.5-VL）。"""
import torch


def image_token_indices(input_ids, image_token_id):
    """Row positions in the flattened prompt occupied by image tokens."""
    return torch.argwhere(input_ids == image_token_id).squeeze(1)


def values_to_grid(values, grid_thw):
    """Reshape per-image-token values [n] to a spatial grid.

    Args:
        values: tensor [n] — per-image-token values, row-major (vision output order).
        grid_thw: sequence of 3 ints (t, h, w) with t*h*w == n.

    Returns:
        grid: tensor [h, w] (single image, t == 1), or [t, h, w] (video).
    """
    t, h, w = (int(x) for x in grid_thw)
    n = t * h * w
    assert values.numel() == n, f"{values.numel()} values but grid is {n}"
    grid = values.reshape(t, h, w)
    if t == 1:
        return grid[0]
    return grid
