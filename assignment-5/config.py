from easydict import EasyDict

cfg = EasyDict()

cfg.img_path = 'emma.jpg'  # 'valve.png'

cfg.GaussianBlurFS = 11  #7 # 11  # 11 7
cfg.GaussianBlurStd = 0  # 1 1.4

cfg.low_threshold = 0.8
cfg.high_threshold = 0.2
