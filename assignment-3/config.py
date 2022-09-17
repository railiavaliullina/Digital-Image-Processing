from easydict import EasyDict

cfg = EasyDict()

cfg.path_to_img = 'Houses.jpg'

# sauvola_binarization params
cfg.w = [14, 60]
cfg.r = 128
cfg.k = 0.2  # [0.2, 0.5]
