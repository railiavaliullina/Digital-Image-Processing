from easydict import EasyDict

cfg = EasyDict()
cfg.path_to_img = 'segment.jpg'
cfg.lower_hsv = [30, 0, 0]  # [30, 0, 0]
cfg.upper_hsv = [107, 255, 232]  # [107, 255, 232]  # [109, 255, 255]

cfg.load_saved_contours = False # False
cfg.size_lower_threshold_cff = 0.663674  # 0.65
cfg.size_upper_threshold_cff = 1.0
cfg.bb_color = (0, 0, 255)
