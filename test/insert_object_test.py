#!/usr/bin/env python3
import os
import sys
import unittest
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from basic_approaches.basic_img_augmentation import (
    data_loader,
    obj_preprocesser,
    person_value,
)
from utils.geometric_transformations import (
    person_height_calculation,
    obj_resizer,
    fg_bg_preprocesser,
)
from utils.datastructures import Pixel


class TestHeightEstimation(unittest.TestCase):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    bg_files_path = os.path.abspath(os.path.join(file_dir, "../files/"))
    fg_files_path = os.path.abspath(os.path.join(file_dir, "../files/"))
    bg_dict = {
        "img": os.path.join(fg_files_path, "jena_000092_000019_leftImg8bit.png"),
        "mask": os.path.join(fg_files_path, "jena_000092_000019_gtFine_labelIds.png"),
        "camera": os.path.join(fg_files_path, "jena_000092_000019_camera.json"),
    }
    fg_dict = {
        "img": os.path.join(bg_files_path, "jena_000075_000019_leftImg8bit.png"),
        "mask": os.path.join(bg_files_path, "jena_000075_000019_gtFine_labelIds.png"),
        "camera": os.path.join(bg_files_path, "jena_000075_000019_camera.json"),
    }
    fg_img, fg_mask, bg_img, bg_mask, camera_dict = data_loader(fg_dict, bg_dict)
    bg_height = bg_img.shape[0]
    bg_width = bg_img.shape[1]
    fg_height = fg_img.shape[0]
    fg_width = fg_img.shape[1]
    obj_img, obj_mask, obj_rect = obj_preprocesser(fg_img, fg_mask, person_value)
    # define positions where occlusion is guaranteed
    test_runs = [{"x": 136, "y": 660}, {"x": 1429, "y": 602}, {"x": 1629, "y": 886}]

    def test_(self):
        for test in self.test_runs:
            # get basepoint for pedestrian
            person_x = test["x"]
            person_y = test["y"]
            # extract the rectangle where the object will be placed

            person_height = round(
                person_height_calculation(self.camera_dict, person_x, person_y)
            )
            person_width = round(self.obj_rect.w / self.obj_rect.h * person_height)
            resized_obj_img, resized_obj_mask = obj_resizer(
                self.obj_img, self.obj_mask, person_height, person_width, person_value
            )
            fg_bg_img, fg_bg_mask, alpha_mask = fg_bg_preprocesser(
                resized_obj_img,
                resized_obj_mask,
                self.bg_img,
                self.bg_mask,
                Pixel(person_x, person_y),
                person_height,
                person_width,
                person_value,
            )
            plt.imshow(fg_bg_img)
            plt.show()


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
