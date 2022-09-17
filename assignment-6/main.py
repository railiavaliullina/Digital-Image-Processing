import numpy as np
from utils import get_img, show_img, fourier_transform, get_rotation_angle, get_rotated_img
from config import cfg


if __name__ == '__main__':
    source_img = get_img(cfg.img_path)

    start_angles = [27, 71, 158, -180, -135, -100, -90, -45, -32, 0, 37, 45, 90, 100, 135, 180]
    # start_angles = np.random.choice(np.arange(-180, 181), 10)
    for ang in start_angles:
        print('\nstart rotation on angle: ', ang)
        source_img_ = get_rotated_img(source_img, ang)
        img = fourier_transform(source_img_)
        angle = get_rotation_angle(img)
        res_image = get_rotated_img(source_img_, angle)
        show_img((source_img_, res_image), subplots=True, title=(f'start image, rotated on {ang} degrees',
                                                                 f'result image'))
