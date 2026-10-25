import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from token_mapping import image_token_indices, values_to_grid


def test_image_token_indices():
    input_ids = torch.tensor([1, 151655, 151655, 151655, 2])  # image_token_id = 151655
    idx = image_token_indices(input_ids, 151655)
    assert idx.tolist() == [1, 2, 3]
    print("indices OK")


def test_values_to_grid():
    grid_thw = (1, 16, 16)
    v = torch.arange(256.0)
    g = values_to_grid(v, grid_thw)
    assert tuple(g.shape) == (16, 16)
    assert g[0, 0].item() == 0.0 and g[15, 15].item() == 255.0
    print("grid OK")


def test_mismatch_raises():
    try:
        values_to_grid(torch.ones(100), (1, 16, 16))
        raise SystemExit("should have raised AssertionError")
    except AssertionError:
        print("mismatch raises OK")


if __name__ == "__main__":
    test_image_token_indices()
    test_values_to_grid()
    test_mismatch_raises()
    print("ALL PASS")
