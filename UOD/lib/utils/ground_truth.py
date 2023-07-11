import numpy as np


def gen_heatmap(image_size,coords,sigma,alpha, dtype=np.float32):

    heatmap = np.zeros(image_size, dtype=dtype)
    size_sigma_factor = 10

    # flip point from [x, y, z] to [z, y, x]
    flipped_coords = np.flip(coords, 0)
    region_start = (flipped_coords - sigma * size_sigma_factor / 2).astype(int)
    region_end = (flipped_coords + sigma * size_sigma_factor / 2).astype(int)
    region_start = np.maximum(0, region_start).astype(int)
    region_end = np.minimum(image_size, region_end).astype(int)

    # return zero landmark, if region is invalid, i.e., landmark is outside of image
    if np.any(region_start >= region_end):
        return heatmap

    region_size = (region_end - region_start).astype(int)

    dy, dx = np.meshgrid(range(region_size[1]), range(region_size[0]))
    x_diff = dx + region_start[0] - flipped_coords[0]
    y_diff = dy + region_start[1] - flipped_coords[1]

    squared_distances = x_diff * x_diff + y_diff * y_diff

    cropped_heatmap = np.exp(-squared_distances / (2 * (sigma** 2)))

    heatmap[region_start[0]:region_end[0],
    region_start[1]:region_end[1]] = cropped_heatmap[:, :]

    #
    heatmap = np.power(alpha, heatmap)
    heatmap[heatmap <= 1.05] = 0

    return heatmap
