import numpy as np
from scipy.ndimage.measurements import label as scipy_label
from scipy.ndimage.measurements import center_of_mass


def get_landmark_by_component(heatmaps):
    nums = heatmaps.shape[0]
    pred = np.zeros((nums, 2))

    for i in range(nums):
        heatmap = heatmaps[i]
        if np.max(heatmap) <= 0:
            heatmap[heatmap < 1.25 * np.max(heatmap)] = 0
            heatmap = -heatmap
        else:
            heatmap[heatmap<0.25*np.max(heatmap)] = 0
        structure = np.ones((3, 3), dtype=np.int)
        components_labels, ncomponents = scipy_label(heatmap, structure)

        count_max = 0
        label = 0
        for l in range(1, ncomponents + 1):
            component = (components_labels == l)
            count = np.count_nonzero(component) 

            if count > count_max:
                count_max = count
                label = l

        heatmap[components_labels != label] = 0
        y_array, x_array = np.where(heatmap > 0.88 * np.max(heatmap))
        pred[i] = np.array([x_array, y_array]).mean(axis=-1)

    return pred


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def get_landmark_by_argmax(arr):
    ''' 
        arr: numpy.ndarray, channel x imageshape
        ret: [(x,y..)]* channel
    '''

    points = []
    for img in arr:
        index = img.argmax()
        points.append(unravel_index(index, img.shape))
    return points
