import cv2
import time
import numpy as np
from copy import deepcopy
from skimage.data import page
from config import cfg


def show_img(img, name='image', size=None):
    img = cv2.resize(img, size) if size is not None else img
    cv2.imshow(name, img)
    cv2.waitKey(0)


def threshold_binarization(img, thr=150):
    if not isinstance(img, int):
        img[img >= thr] = 255
        img[img < thr] = 0
    else:
        img = 255 if img >= thr else 0
    return img


def otsu_binarization(img):
    start_time = time.time()
    p, _ = np.histogram(img, bins=256)
    fin_thr, fin_sigma = 0, 0
    i = np.arange(256)
    thresholds = i[1:]
    for thr in thresholds:
        nB = np.sum(p[:thr])
        nO = np.sum(p[thr:])
        mB = np.sum(i[:thr] * p[:thr]) / nB
        mO = np.sum(i[thr:] * p[thr:]) / nO

        sigma_between = nB*nO*(mB - mO)**2
        if sigma_between > fin_sigma:
            fin_sigma = sigma_between
            fin_thr = thr
    img = threshold_binarization(img, thr=fin_thr)
    print(f'otsu binarization time: {round((time.time() - start_time) / 60, 6)} min')
    return img


def get_integral_img(img):
    img = img.astype('float64')
    h, w = img.shape
    integral_img = np.zeros_like(img, dtype='float64')
    for i in range(h):
        for j in range(w):
            if i > 0 and j > 0:
                integral_img[i, j] = img[i, j]-integral_img[i-1, j-1]+integral_img[i, j-1]+integral_img[i-1, j]
            elif i > 0:
                integral_img[i, j] = img[i, j]+integral_img[i-1, j]
            elif j > 0:
                integral_img[i, j] = img[i, j]+integral_img[i, j-1]
            else:
                integral_img[i, j] = img[i, j]
    return integral_img


def sauvola_binarization(img, w_size=cfg.w, k=cfg.k, r=cfg.r, pad_image=False):
    start_time = time.time()
    win_h, win_w = w_size
    if pad_image:
        img = np.pad(img, ((win_h//2, win_h//2), (win_w//2, win_w//2)), mode='reflect')

    img_h, img_w = img.shape
    integral_img = get_integral_img(img)
    img_ = img.astype(int)**2
    integral_img_sq = get_integral_img(img_)

    start_i, fin_i = win_h//2 if pad_image else 0, img_h - win_h//2 if pad_image else img_h
    start_j, fin_j = win_w//2 if pad_image else 0, img_w - win_w//2 if pad_image else img_w
    for i in range(start_i, fin_i):
        i0, i1 = i - win_h//2, i + win_h//2
        if not pad_image:
            i0, i1 = np.clip(i0, a_min=0, a_max=None), np.clip(i1, a_max=img_h-1, a_min=None)
        for j in range(start_j, fin_j):
            j0, j1 = j - win_w//2, j + win_w//2
            if not pad_image:
                j0, j1 = np.clip(j0, a_min=0, a_max=None), np.clip(j1, a_max=img_w-1, a_min=None)
            s1 = integral_img[i0, j0] + integral_img[i1, j1] - integral_img[i0, j1] - integral_img[i1, j0]
            n = (i1 - i0)*(j1 - j0)
            m = s1/n
            s2 = integral_img_sq[i0, j0] + integral_img_sq[i1, j1] - integral_img_sq[i0, j1] - integral_img_sq[i1, j0]
            std = np.sqrt(s2/n - m**2)
            thr = m*(1 + k * ((std / r) - 1))
            bin_win = threshold_binarization(int(img[i, j]), thr)
            img[i, j] = bin_win

    if pad_image:
        img = img[win_h//2:img_h - win_h//2, win_w//2:img_w - win_w//2]
    print(f'sauvola binarization time: {round((time.time() - start_time)/60, 3)} min')
    return img


if __name__ == '__main__':
    source_img = cv2.imread(cfg.path_to_img)
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    # show_img(source_img)

    bin_by_thr_img = threshold_binarization(deepcopy(source_img))
    # show_img(bin_by_thr_img, 'threshold binarization result')

    bin_by_otsu_img = otsu_binarization(deepcopy(source_img))
    cv2.imwrite('otsu_result(houses).png', np.array(bin_by_otsu_img))
    # show_img(bin_by_otsu_img, 'otsu binarization result')

    image = page()
    bin_by_otsu_img2 = otsu_binarization(deepcopy(image))
    cv2.imwrite('otsu_result(page).png', np.array(bin_by_otsu_img2))
    # show_img(bin_by_otsu_img2, 'otsu binarization result')

    bin_by_sauvola_img = sauvola_binarization(deepcopy(source_img))
    cv2.imwrite('sauvola_result(houses).png', np.array(bin_by_sauvola_img))
    # show_img(bin_by_sauvola_img, 'sauvola binarization result')

    bin_by_sauvola_img_padded = sauvola_binarization(deepcopy(source_img), pad_image=True)
    cv2.imwrite('sauvola_result(houses, padded_image).png', np.array(bin_by_sauvola_img))
    # show_img(bin_by_sauvola_img_padded, 'sauvola binarization result(padded img)')

    bin_by_sauvola_img2 = sauvola_binarization(deepcopy(image))
    cv2.imwrite('sauvola_result(page).png', np.array(bin_by_sauvola_img2))
    # show_img(bin_by_sauvola_img2, 'sauvola binarization result2')

    bin_by_sauvola_img2_padded = sauvola_binarization(deepcopy(image), pad_image=True)
    cv2.imwrite('sauvola_result(page, padded_image).png', np.array(bin_by_sauvola_img2))
    # show_img(bin_by_sauvola_img2_padded, 'sauvola binarization result2(padded img)')
