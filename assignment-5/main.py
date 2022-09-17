import cv2
import numpy as np
import time
from copy import deepcopy

from utils import show_img
from config import cfg


class Set:
    def __init__(self, label):
        self.label = label
        self.rank = 0
        self.root = self


class Sets:
    def __init__(self):
        self.sets = {}

    def make_set(self, label):
        set_ = Set(label)
        self.sets.update({label: set_})
        return set

    def union(self, set0, set1):
        root0, root1 = self.find(set0), self.find(set1)
        rank0, rank1 = root0.rank, root1.rank
        if rank0 == rank1:
            root1.root = root0
            root0.rank += 1
        elif rank0 > rank1:
            root1.root = root0
            root0.rank = np.max([rank0, rank1])
        else:
            root0.root = root1
            root1.rank = np.max([rank0, rank1])

    def find(self, set):
        set.root = self.find(set.root) if set.root != set else set.root
        return set.root


def apply_sobel_filter(img):
    x_filter = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    y_filter = np.rot90(x_filter, axes=(1, 0))
    g_x = cv2.filter2D(src=img, kernel=x_filter, ddepth=-1, borderType=cv2.BORDER_REFLECT)
    g_y = cv2.filter2D(src=img, kernel=y_filter, ddepth=-1, borderType=cv2.BORDER_REFLECT)
    g = np.sqrt(g_x ** 2 + g_y ** 2)
    theta = np.arctan2(g_y, g_x)
    theta[theta < 0] += np.pi
    return g, theta


def apply_non_maximum_suppression(g, theta):
    g = np.pad(g, ((1, 1), (1, 1)), mode='reflect')
    theta = np.pad(theta, ((1, 1), (1, 1)), mode='reflect')
    h, w = g.shape
    g_res = np.zeros((h, w))
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if g[r, c] > 0:
                if (0 <= theta[r, c] < np.pi / 8) or (np.pi - np.pi / 8 <= theta[r, c] <= np.pi):
                    win = g[r, c - 1:c + 2]
                elif np.pi / 8 <= theta[r, c] < np.pi / 2 - np.pi / 8:
                    win = [g[r - 1, c - 1], g[r, c], g[r + 1, c + 1]]
                elif np.pi / 2 - np.pi / 8 <= theta[r, c] < np.pi / 2 + np.pi / 8:
                    win = g[r - 1: r + 2, c]
                else:
                    win = [g[r + 1, c - 1], g[r, c], g[r - 1, c + 1]]
                g_res[r, c] = g[r, c] if g[r, c] == max(win) else 0
    g_res = g_res[1:h - 1, 1:w - 1]
    return g_res


def connected_components_labeling(img):
    Sets_ = Sets()
    last_label = 1
    img[img < cfg.low_threshold] = 0
    img = np.pad(img, ((1, 1), (1, 1)), mode='reflect')
    h, w = img.shape
    labels = np.zeros((h, w))
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if img[r, c] > 0:
                neighbors_labels = labels[r - 1: r + 2, c - 1:c + 2]
                non_zero_neighbors = neighbors_labels[neighbors_labels != 0]

                if non_zero_neighbors.any():
                    min_label = np.min(non_zero_neighbors)
                    labels[r, c] = min_label
                    unique_neighbors = list(set(non_zero_neighbors))
                    unique_neighbors = np.delete(unique_neighbors, np.argmin(unique_neighbors))

                    if unique_neighbors.any():
                        unique_neighbors = unique_neighbors[unique_neighbors != min_label]
                        set0 = Sets_.sets[min_label]
                        for label in unique_neighbors:
                            set1 = Sets_.sets[label]
                            if set0.root != set1.root:
                                Sets_.union(set0, set1)
                else:
                    labels[r, c] = last_label
                    Sets_.make_set(last_label)
                    last_label += 1

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if img[r, c] > 0:
                root = Sets_.find(Sets_.sets[labels[r, c]])
                labels[r, c] = root.label

    labels = labels[1:h - 1, 1:w - 1]
    return labels


def hysteresis_threshold(img):
    cfg.high_threshold *= img.max()
    cfg.low_threshold *= cfg.high_threshold
    img[img < cfg.low_threshold] = 0
    x_strong, y_strong = np.where((img >= cfg.high_threshold))
    strong_coord = np.stack([x_strong, y_strong], 1)
    labels = connected_components_labeling(nms_image)
    edges = np.unique(labels)
    edges = edges[edges != 0]
    for l in edges:
        x, y = np.where(labels == l)
        coord = np.stack([x, y], 1)
        if coord.any() in strong_coord:
            img[x, y] = 255
        else:
            img[x, y] = 0
    return img


if __name__ == "__main__":
    source_img = cv2.imread(cfg.img_path).astype('float32')
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    start_time = time.time()

    blured_image = cv2.GaussianBlur(source_img, (cfg.GaussianBlurFS, cfg.GaussianBlurFS), cfg.GaussianBlurStd)
    g, theta = apply_sobel_filter(blured_image)
    nms_image = apply_non_maximum_suppression(g, theta)
    # show_img(nms_image)
    # cv2.imwrite('nms_image.jpg', nms_image)
    res_image = hysteresis_threshold(nms_image)
    # show_img(res_image)
    cv2.imwrite('result_image.jpg', res_image)

    print(f'Total time: {round((time.time() - start_time) / 60, 3)} min')
