class Pixel:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Rectangle:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

class AugmentationWorkerParams:
    def __init__(self, save_directory, fg_path_list, bg_path_list, force_occlusion_flag, min_occlusion_ratio, annotat_status):
        self.save_directory = save_directory
        self.fg_path_list = fg_path_list
        self.bg_path_list = bg_path_list
        self.force_occlusion_flag = force_occlusion_flag
        self.min_occlusion_ratio = min_occlusion_ratio
        self.annotat_status = annotat_status