#!/usr/bin/env python3
import os
import sys
import unittest
import cv2
from matplotlib import pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from basic_approaches.basic_img_augmentation import (
    data_loader,
    obj_preprocesser,
    person_value,
)
from basic_approaches.geometric_transformations import (
    person_size_finder,
    obj_resizer,
    fg_bg_preprocesser,
)


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
    obj_img, obj_mask, x, y, w, h = obj_preprocesser(
        fg_img, fg_mask, bg_height, bg_width, person_value, fg_height, fg_width
    )
    # define positions where occlusion is guaranteed
    test_runs = [{"x": 136, "y": 660}, {"x": 1429, "y": 602}, {"x": 1629, "y": 886}]

    def test_(self):
        for test in self.test_runs:
            # get basepoint for pedestrian
            person_x = test["x"]
            person_y = test["y"]
            # extract the rectangle where the object will be placed
            obj_mask_height = self.obj_mask.shape[0]
            obj_mask_width = self.obj_mask.shape[1]

            stand_obj_height, stand_obj_width = person_size_finder(
                person_y, self.w, self.h
            )
            resized_obj_img, resized_obj_mask = obj_resizer(
                self.obj_img,
                self.obj_mask,
                stand_obj_height,
                stand_obj_width,
                person_value,
            )
            fg_bg_img, fg_bg_mask, alpha_mask = fg_bg_preprocesser(
                resized_obj_img,
                resized_obj_mask,
                self.bg_img,
                self.bg_mask,
                person_x,
                person_y,
                stand_obj_height,
                stand_obj_width,
                self.bg_height,
                self.bg_width,
                person_value,
            )
            plt.imshow(fg_bg_img)
            plt.show()


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
