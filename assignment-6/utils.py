import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rotate, hough_line, hough_line_peaks
import cv2
from config import cfg


def get_img(path):
    return cv2.cvtColor(cv2.imread(path).astype('float32'), cv2.COLOR_BGR2GRAY)


def show_img(img, scale=True, title='', subplots=False):
    if not subplots:
        if scale:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        title1, title2 = title
        ax1.set_title(title1)
        ax2.set_title(title2)
        img1, img2 = img
        ax1.imshow(img1, cmap='gray')
        ax2.imshow(img2, cmap='gray')
    plt.show()


def fourier_transform(img):
    f = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    f = np.fft.fftshift(f)
    r, im = f[:, :, 0], f[:, :, 1]
    p = np.log(cv2.magnitude(r, im))
    # show_img(p, scale=False)
    p[p < cfg.threshold * np.max(p)] = 0
    # show_img(p, scale=False)
    return p


def get_rotation_angle(image_edges):
    out, angles, d = hough_line(image_edges, theta=np.deg2rad(np.arange(-180, 181)))
    # show_img(out, scale=False)
    accum, angles, dists = hough_line_peaks(out, angles, d)
    max_accum_idxs = np.where(accum == np.max(accum))
    print('angles to choose: ', np.rad2deg(angles[max_accum_idxs]))
    angle = np.rad2deg(angles[np.argmax(accum)])
    print('chosen angle: ', angle)
    return angle


def get_rotated_img(img, angle):
    return rotate(img, angle)
