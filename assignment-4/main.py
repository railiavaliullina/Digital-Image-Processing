import cv2
import time
import numpy as np
import warnings
from scipy.ndimage import median_filter

from config import cfg
from utils import show_img
warnings.filterwarnings('ignore')


def get_preprocessed_g(x_filter, y_filter, img):
    g_x = abs(cv2.filter2D(src=img, kernel=x_filter, ddepth=-1, borderType=cv2.BORDER_REFLECT))
    g_y = abs(cv2.filter2D(src=img, kernel=y_filter, ddepth=-1, borderType=cv2.BORDER_REFLECT))
    g = np.sqrt(g_x**2 + g_y**2)
    div = g_y/g_x
    div[np.isnan(div)] = 0
    div[np.isinf(div)] = g_y[np.isinf(div)]
    theta = np.arctan(div) + np.pi/2
    cfg.theta_threshold = np.radians(cfg.theta_threshold)
    cond1 = (0.0 <= theta) * (theta <= cfg.theta_threshold)
    cond2 = (np.pi/2 - cfg.theta_threshold <= theta) * (theta <= np.pi/2 + cfg.theta_threshold)
    cond3 = (np.pi - cfg.theta_threshold <= theta) * (theta <= np.pi)
    res_mask = cond1 + cond2 + cond3
    g_x, g_y = g_x * res_mask, g_y * res_mask
    g_x[g > cfg.g_threshold] = 0.0
    g_y[g > cfg.g_threshold] = 0.0
    return g_x, g_y


def get_grid_lines(g, lines_type=''):
    print(f'extracting grid lines of type: {lines_type}..')
    if lines_type == 'v':
        g = g.T
    e_s = cv2.filter2D(src=g, kernel=np.ones((1, cfg.ac)), ddepth=-1, borderType=cv2.BORDER_REFLECT)
    medians = median_filter(e_s, size=(cfg.ac, 1))
    e = e_s - medians
    e[e < 0] = 0
    offsets = np.arange(0, cfg.ac, 8)
    footprint = np.zeros((cfg.ac, 1))
    footprint[offsets] = 1.0
    g_res = median_filter(e, footprint=footprint)
    if lines_type == 'v':
        g_res = g_res.T
    return g_res


if __name__ == '__main__':
    source_img = cv2.imread(cfg.img_path).astype('float32')
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    start_time = time.time()

    x_filter = np.array([[-1, 2, -1]])
    y_filter = x_filter.T
    g_x, g_y = get_preprocessed_g(x_filter, y_filter, source_img)
    g_h = get_grid_lines(g_y, lines_type='h')
    g_v = get_grid_lines(g_x, lines_type='v')
    g = g_h + g_v

    h, w = g.shape
    for r in range(0, h - cfg.block_size, cfg.block_size):
        for c in range(0, w - cfg.block_size, cfg.block_size):
            block = g[r:r + cfg.block_size, c:c + cfg.block_size]
            b1 = np.sum(block[1:cfg.block_size - 1, 1:cfg.block_size - 1], 0)
            b2 = np.sum([block[0, 1:cfg.block_size - 1], block[-1, 1:cfg.block_size - 1]], 1)
            b3 = np.sum(block[1:cfg.block_size - 1, 1:cfg.block_size - 1], 1)
            b4 = np.sum([block[1:cfg.block_size - 1, 0], block[1:cfg.block_size - 1, -1]], 1)
            b = np.max(b1) - np.min(b2) + np.max(b3) - np.min(b4)
            g[r:r + cfg.block_size, c:c + cfg.block_size] = np.ones((cfg.block_size, cfg.block_size)) * b
    show_img(g)
    cv2.imwrite('anomaly_score.png', np.array(g))
    print(f'Total time: {round((time.time() - start_time) / 60, 3)} min')
