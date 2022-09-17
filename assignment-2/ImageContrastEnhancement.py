import cv2
import numpy as np
from copy import deepcopy
import math
from configHW2 import cfg


def show_img(img, name='image', size=None):
    img = cv2.resize(img, size) if size is not None else img
    cv2.imshow(name, img)
    cv2.waitKey(0)


def histogram_equalization(img, clip_l=None):  # клипирование не доделано, не разобралась с багом
    hist, bins = np.histogram(img.flatten(), bins=256)
    if clip_l is not None:
        r = len(hist)
        top = clip_l
        bottom, s = 0, 0
        while top-bottom > 1:
            mid = (top+bottom)/2
            a = [p for p in hist if p > mid]
            s = np.sum(a)
            if s > (clip_l - mid)*r:
                top = mid
            else:
                bottom = mid
        bottom = round(bottom)
        p = bottom + s/r
        l = clip_l - p
        hist_ = [None for _ in range(len(hist))]
        for i in range(len(hist)):
            if hist[i] < p:
                hist_[i] = hist[i] + l
            else:
                hist_[i] = clip_l
        hist = hist_
    pdf = hist / np.sum(hist)
    cdf = np.cumsum(pdf)
    cdf = np.array([int(round(c)) for c in cdf / np.float(cdf[-1]) * 255])
    res_img = np.reshape(cdf[img.flatten()], img.shape).astype(np.uint16)
    return res_img, cdf


def get_splitted_image(img):
    h, w = img.shape
    h_squares_pixels_num, w_squares_pixels_num = int(math.floor(h / cfg.squares_num)), int(
        math.floor(w / cfg.squares_num))
    h_mod_square_num, w_mod_square_num = h % cfg.squares_num,  w % cfg.squares_num

    splitted_image_ = [[[] for _ in range(cfg.squares_num)] for _ in range(cfg.squares_num)]
    checkpoints_ = []
    checkpoints_ids = [[[] for _ in range(cfg.squares_num)] for _ in range(cfg.squares_num)]
    lookup_tables_ = [[[] for _ in range(cfg.squares_num)] for _ in range(cfg.squares_num)]
    for i in range(cfg.squares_num):
        i1, i2 = i * h_squares_pixels_num, (i + 1) * h_squares_pixels_num
        i2 += h_mod_square_num if i == cfg.squares_num - 1 else 0
        cur_line = img[i1:i2, :]
        i_center = i2 - int(round((i2 - i1) / 2))
        for j in range(cfg.squares_num):
            j1, j2 = j * w_squares_pixels_num, (j + 1) * w_squares_pixels_num
            j2 += w_mod_square_num if j == cfg.squares_num - 1 else 0
            splitted_image_[i][j] = cur_line[:, j1:j2]
            _, lookup_tables_[i][j] = histogram_equalization(np.array(splitted_image_[i][j]))
            j_center = j2 - int(round((j2 - j1) / 2))
            checkpoints_.append([i_center, j_center])
            checkpoints_ids[i][j] = [i_center, j_center]
    return splitted_image_, (checkpoints_, checkpoints_ids), lookup_tables_


if __name__ == '__main__':
    source_img = cv2.imread(cfg.path_to_img)
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    source_img_ = deepcopy(source_img)
    source_img_min_val, source_img_max_val = np.min(source_img), np.max(source_img)
    # show_img(source_img)

    contextual_regions, checkpoints, lookup_tables = get_splitted_image(source_img)
    xs_ckpt, ys_ckpt = np.array(checkpoints[0])[:, 0], np.array(checkpoints[0])[:, 1]
    min_x_ckpt, max_x_ckpt, min_y_ckpt, max_y_ckpt = min(xs_ckpt), max(xs_ckpt), min(ys_ckpt), max(ys_ckpt)

    blocks, v_border_blocks, h_border_blocks = [], [], []
    l_tables, l_tables_corn, v_border_tables, h_border_tables = [], [], [], []
    a = checkpoints[1]
    for i in range(len(a)):
        for j in range(len(a)):
            if i in range(1, len(a)) and j in range(1, len(a)):
                blocks.append([a[i-1][j-1], a[i-1][j], a[i][j], a[i][j-1]])
                l_tables.append([lookup_tables[i-1][j-1], lookup_tables[i-1][j], lookup_tables[i][j], lookup_tables[i][j-1]])
            if a[i][j][0]==min_x_ckpt and a[i][j][1]==min_y_ckpt:
                cur_img_block = source_img[:min_x_ckpt, :min_y_ckpt]
                res_img_block = lookup_tables[i][j][cur_img_block]
                up_l_corner_block = res_img_block
            if a[i][j][0]==min_x_ckpt and a[i][j][1]==max_y_ckpt:
                cur_img_block = source_img[:min_x_ckpt, max_y_ckpt:]
                res_img_block = lookup_tables[i][j][cur_img_block]
                up_r_corner_block = res_img_block
            if a[i][j][0]==max_x_ckpt and a[i][j][1]==min_y_ckpt:
                cur_img_block = source_img[max_x_ckpt:, :min_y_ckpt]
                res_img_block = lookup_tables[i][j][cur_img_block]
                l_l_corner_block = res_img_block
            if a[i][j][0]==max_x_ckpt and a[i][j][1]==max_y_ckpt:
                cur_img_block = source_img[max_x_ckpt:, max_y_ckpt:]
                res_img_block = lookup_tables[i][j][cur_img_block]
                l_r_corner_block = res_img_block
            # vertical borders
            if (a[i][j][1]==min_y_ckpt or a[i][j][1]==max_y_ckpt) and a[i][j][0] < max_x_ckpt:
                v_border_blocks.append([a[i][j], a[i+1][j]])
                v_border_tables.append([lookup_tables[i][j], lookup_tables[i+1][j]])
            # horizontal borders
            if (a[i][j][0] == min_x_ckpt or a[i][j][0] == max_x_ckpt) and a[i][j][1] < max_y_ckpt:
                h_border_blocks.append([a[i][j], a[i][j+1]])
                h_border_tables.append([lookup_tables[i][j], lookup_tables[i][j+1]])

    all_blocks = [[] for _ in range(cfg.squares_num-1)]  # blocks of blue zone
    k = -1
    for b_id, b in enumerate(blocks):
        xs, ys = np.array(b)[:, 0], np.array(b)[:, 1]
        min_bx, max_bx, min_by, max_by = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
        cur_img_block = source_img[min_bx:max_bx, min_by:max_by]
        lts = l_tables[b_id]
        up_l_lt, up_r_lt, low_r_lt, low_l_lt = lts[0], lts[1], lts[2], lts[3]

        for i in range(cur_img_block.shape[0]):
            for j in range(cur_img_block.shape[1]):
                pxl = cur_img_block[i][j]
                up_l_map, up_r_map, low_r_map, low_l_map = up_l_lt[pxl], up_r_lt[pxl], low_r_lt[pxl], low_l_lt[pxl]
                a_const = (min_by+j - min_by) / (max_by - min_by)
                b_const = (min_bx+i - min_bx) / (max_bx - min_bx)
                # res = (1-a_const)*(1-b_const)*up_l_map + b_const*(1-a_const)*low_l_map + (1-b_const)*a_const*up_r_map+a_const*b_const*low_r_map
                res = (1-a_const)*(1-b_const)*up_l_map + b_const*(1-a_const)*low_l_map + (1-b_const)*a_const*up_r_map+a_const*b_const*low_r_map
                cur_img_block[i][j] = res
        if b_id % (cfg.squares_num-1) == 0:
            k+=1
        all_blocks[k].append(cur_img_block)
    for i in range(len(all_blocks)):
        all_blocks[i] = np.concatenate(all_blocks[i], 1)
    blue_zone_img=np.concatenate(all_blocks, 0)

    # get borders
    # vertical
    l_v_blocks, r_v_blocks = [], []
    for vb_id, vb in enumerate(v_border_blocks):
        xs, ys = np.array(vb)[:, 0], np.array(vb)[:, 1]
        min_vbx, max_vbx, min_vby = np.min(xs), np.max(xs), np.min(ys)
        lts = v_border_tables[vb_id]
        up_lt, low_lt = lts[0], lts[1]
        if min_vby == min_y_ckpt:
            cur_img_block = source_img[min_vbx:max_vbx, :min_y_ckpt]
            type = 'l'
        else:
            cur_img_block = source_img[min_vbx:max_vbx, max_y_ckpt:]
            type = 'r'
        for i in range(cur_img_block.shape[0]):
            for j in range(cur_img_block.shape[1]):
                pxl = cur_img_block[i][j]
                up_m, low_m = up_lt[pxl], low_lt[pxl]
                a_c = (min_vbx+i - min_vbx)/(max_vbx - min_vbx)
                res = a_c * low_m + (1-a_c) * up_m
                cur_img_block[i][j] = res
        if type == 'l':
            l_v_blocks.append(cur_img_block)
        else:
            r_v_blocks.append(cur_img_block)

    # horizontal
    u_h_blocks, l_h_blocks = [], []
    for hb_id, hb in enumerate(h_border_blocks):
        xs, ys = np.array(hb)[:, 0], np.array(hb)[:, 1]
        min_hbx, min_hby, max_hby = np.min(xs), np.min(ys), np.max(ys)
        lts = h_border_tables[hb_id]
        l_lt, r_lt = lts[0], lts[1]
        if min_hbx == min_x_ckpt:
            cur_img_block = source_img[:min_x_ckpt, min_hby:max_hby]
            type = 'u'
        else:
            cur_img_block = source_img[max_x_ckpt:, min_hby:max_hby]
            type = 'l'
        for i in range(cur_img_block.shape[0]):
            for j in range(cur_img_block.shape[1]):
                pxl = cur_img_block[i][j]
                up_m, low_m = l_lt[pxl], r_lt[pxl]
                a_c = (min_vbx+i - min_vbx)/(max_vbx - min_vbx)
                # a_c, b_c = (max_hby - i)/(max_hby - min_hby), (i - min_hby)/(max_hby - min_hby)
                res = (1-a_c) * up_m + a_c * low_m
                cur_img_block[i][j] = res
        if type == 'u':
            u_h_blocks.append(cur_img_block)
        else:
            l_h_blocks.append(cur_img_block)

    l_v, r_v = np.concatenate(l_v_blocks), np.concatenate(r_v_blocks)
    u_h, l_h = np.concatenate(u_h_blocks, 1), np.concatenate(l_h_blocks, 1)
    res = np.concatenate([l_v, blue_zone_img, r_v], axis=1)
    u_h = np.concatenate([up_l_corner_block, u_h, up_r_corner_block], axis=1)
    l_h = np.concatenate([l_l_corner_block, l_h, l_r_corner_block], axis=1)
    res = np.concatenate([u_h, res, l_h], axis=0)
    # res = (res - np.min(res))/ (np.max(res) - np.min(res)) * 255
    show_img(res.astype('uint8'))
