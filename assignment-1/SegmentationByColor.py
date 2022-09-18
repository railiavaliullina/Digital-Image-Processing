import cv2
import numpy as np
import time
import pickle
from copy import deepcopy
from config import cfg


def convert_to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def filter_objects_by_color_mask(img):
    green_low_hsv = np.array(cfg.lower_hsv)
    green_high_hsv = np.array(cfg.upper_hsv)
    return cv2.inRange(img, green_low_hsv, green_high_hsv)


def get_bin_img(img):
    return img / 255


def look_around(working_pxl, current_pxl, img, clockwise=True):
    i_cur, j_cur = current_pxl
    i_work, j_work = working_pxl
    d = -1 if clockwise else 1
    degrees = 45 * d
    while degrees != 360 * d:
        angle = np.radians(degrees)
        idx1 = int(round(i_cur + np.cos(angle) * (i_work - i_cur) - np.sin(angle) * (j_work - j_cur)))
        idx2 = int(round(j_cur + np.sin(angle) * (i_work - i_cur) + np.cos(angle) * (j_work - j_cur)))
        if 0 <= idx1 < img.shape[0] and 0 <= idx2 < img.shape[1] - 1 and img[idx1][idx2] != 0:
            return idx1, idx2
        if clockwise:
            degrees -= 45
        else:
            degrees += 45
    return None, None


def change_pixel_value(img, coord, nbd):
    i3, j3 = coord
    if img[i3][j3 + 1] == 0:
        img[i3][j3] = -nbd
    elif img[i3][j3 + 1] != 0 and img[i3][j3] == 1:
        img[i3][j3] = nbd
    return img


def counterclockwise_search(working_pxl, current_pxl, img, nbd):
    i4, j4 = look_around(working_pxl, current_pxl, img, clockwise=False)
    img = change_pixel_value(img, current_pxl, nbd)
    return i4, j4, img


def find_objects_contours(img):
    contours_ = []
    nb_rows, nb_cols = img.shape
    i1, j1, i2, j2, i3, j3 = None, None, None, None, None, None
    nbd = 1
    step = 0
    start_time = time.time()
    print(f'Starting contours search..')
    for i in range(nb_rows):
        lnbd = 1
        for j in range(1, nb_cols - 1):
            cur_contour = []
            if step % 100000 == 0:
                print(f'step: {step}, (i, j): {i, j}')
            step += 1

            if img[i][j] == 1 and img[i][j - 1] == 0:
                # border_type = 'outer'
                nbd += 1
                i2, j2 = i, j - 1
            elif img[i][j] >= 1 and img[i][j + 1] == 0:
                # border_type = 'hole'
                nbd += 1
                i2, j2 = i, j + 1
                lnbd = img[i][j] if img[i][j] > 1 else lnbd
            else:
                if img[i][j] != 1:
                    lnbd = abs(img[i][j])
                continue

            # follow the detected border
            if (i1, j1) == (None, None):
                i1, j1 = look_around((i2, j2), (i, j), img)
            if (i1, j1) == (None, None):
                img[i][j] = -nbd
                if img[i][j] != 1:
                    lnbd = abs(img[i][j])
                continue
            cur_contour.append([j1, i1])
            i2, j2 = i1, j1
            i3, j3 = i, j
            cur_contour.append([j3, i3])
            i4, j4, img = counterclockwise_search((i2, j2), (i3, j3), img, nbd)
            if (i4, j4) != (None, None):
                cur_contour.append([j4, i4])
            if (i4, j4) == (i, j) and (i3, j3) == (i1, j1):
                i1, j1 = None, None
                if img[i][j] != 1:
                    lnbd = abs(img[i][j])
                contours_.append(cur_contour)
                continue
            else:
                while (i4, j4) != (i, j) and (i3, j3) != (i1, j1):
                    if (i4, j4) == (None, None):
                        break
                    i2, j2 = i3, j3
                    i3, j3 = i4, j4
                    i4, j4, img = counterclockwise_search((i2, j2), (i3, j3), img, nbd)
                    if (i4, j4) != (None, None):
                        cur_contour.append([j4, i4])
                i1, j1 = None, None
                contours_.append(cur_contour)

    print(f'Time: {round((time.time() - start_time), 3)} sec')
    save_array('result image', img)
    save_array('contours_list', contours_)
    return img, contours_


def get_bb_by_contours(contours_):
    bounding_boxes_ = []
    for cont in contours_:
        bb = cv2.boundingRect(np.array(np.array(cont)))
        bounding_boxes_.append(bb)
    return list(set(bounding_boxes_))


def filter_objects_by_size(bounding_boxes_):
    areas = [bb[2] * bb[3] for bb in bounding_boxes_]
    filtered_bb = []
    for i, bb in enumerate(bounding_boxes_):
        if cfg.size_lower_threshold_cff * max(areas) <= areas[i] <= cfg.size_upper_threshold_cff * max(areas):
            filtered_bb.append(bb)
    return filtered_bb


def get_result_image_with_bb(img, bounding_boxes_):
    for i, bb in enumerate(bounding_boxes_):
        cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[0] + bb[2]), int(bb[1] + bb[3])),
                      color=cfg.bb_color, thickness=3)


def show_img(img, name='image', size=None):
    img = cv2.resize(img, size) if size is not None else img
    cv2.imshow(name, img)
    cv2.waitKey(0)


def save_array(name, img):
    with open(name, 'wb') as f:
        pickle.dump(img, f)


def load_array(name):
    with open(name, 'rb') as f:
        img = pickle.load(f)
    return img


if __name__ == '__main__':
    source_img = cv2.imread(cfg.path_to_img)
    hsv_img = convert_to_hsv(source_img)
    hsv_img_filtered = filter_objects_by_color_mask(hsv_img)
    bin_img = get_bin_img(hsv_img_filtered)

    if cfg.load_saved_contours:
        bin_img = load_array('result image')
        contours = load_array('contours_list')
    else:
        bin_img, contours = find_objects_contours(bin_img)
    # show_img(bin_img, 'bin_img',  (720, 500))

    # get bounding_boxes by contours
    bounding_boxes = get_bb_by_contours(contours)
    # filter bounding_boxes by size
    bb_filtered_by_size = filter_objects_by_size(bounding_boxes)
    source_img_ = deepcopy(source_img)
    # get result image with all bounding boxes
    get_result_image_with_bb(source_img_, bounding_boxes)
    # get result image with filtered bounding boxes
    get_result_image_with_bb(source_img, bb_filtered_by_size)

    # result image with all bounding boxes
    cv2.imwrite('Result_Image_with_all_bb.jpg', np.array(source_img_))
    show_img(source_img_, 'Result_Image_with_all_bb', (720, 500))
    # result image with filtered bounding boxes
    cv2.imwrite('Result_Image_with_filtered_bb.jpg', np.array(source_img))
    show_img(source_img, 'Result_Image_with_filtered_bb', (720, 500))
