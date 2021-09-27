#!/usr/bin/env python3
import os
import sys
import unittest
from matplotlib import pyplot as plt
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from basic_approaches.utils.constants import person_value
from basic_approaches.utils.geometric_transformations import (
    obj_preprocesser,
    person_height_calculation,
    obj_resizer,
    fg_bg_preprocesser,
    get_obj_start_end
)
from basic_approaches.utils.io_functions import fg_data_loader, bg_data_loader
from basic_approaches.utils.datastructures import Pixel


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
    fg_img, fg_mask = fg_data_loader(fg_dict)
    bg_img, bg_mask, camera_dict = bg_data_loader(bg_dict)
    bg_height = bg_img.shape[0]
    bg_width = bg_img.shape[1]
    fg_height = fg_img.shape[0]
    fg_width = fg_img.shape[1]
    obj_img, obj_mask, obj_rect = obj_preprocesser(fg_img, fg_mask, person_value)
    # define positions where occlusion is guaranteed
    test_runs = [{"x": 1429, "y": 602}]
    image_savedir = "/home/roesch/tmp"
    images_to_save = []
    images_to_save.append({"name": "background.png", "img": bg_img})

    def insert_object_to_background(self, obj_img, obj_mask, person_x, person_y, w, h):
        mask_gray = cv2.cvtColor(obj_mask, cv2.COLOR_RGB2GRAY)
        _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
        masked_obj = cv2.bitwise_and(obj_img, obj_img, mask=mask_binary)
        obj_start_x, obj_start_y, obj_end_x, obj_end_y = get_obj_start_end(
            person_x, person_y, w, h
        )
        bg = self.bg_img[obj_start_y:obj_end_y, obj_start_x:obj_end_x].copy()
        import copy

        image = copy.deepcopy(self.bg_img)
        loc = np.where(mask_binary != 0)
        bg[loc] = masked_obj[loc]
        image[obj_start_y:obj_end_y, obj_start_x:obj_end_x] = bg
        return image, masked_obj

    def test_(self):
        for test in self.test_runs:
            # get basepoint for pedestrian
            person_x = test["x"]
            person_y = test["y"]
            # extract the rectangle where the object will be placed
            unscaled_person_img, unscaled_person_mask = self.insert_object_to_background(self.obj_img, self.obj_mask, person_x, person_y, self.obj_rect.w, self.obj_rect.h)
            self.images_to_save.append({"name": "unscaled_person.png", "img": unscaled_person_img})
            self.images_to_save.append({"name": "unscaled_person_mask.png", "img": unscaled_person_mask})

            person_height = round(
                person_height_calculation(self.camera_dict, person_x, person_y)
            )
            person_width = round(self.obj_rect.w / self.obj_rect.h * person_height)
            resized_obj_img, resized_obj_mask = obj_resizer(
                self.obj_img, self.obj_mask, person_height, person_width, person_value
            )
            scaled_person_img, _ = self.insert_object_to_background(resized_obj_img, resized_obj_mask, person_x, person_y, person_width, person_height)
            self.images_to_save.append({"name": "scaled_person.png", "img": scaled_person_img})

            fg_bg_img, fg_bg_mask, alpha_mask, unblended_image = fg_bg_preprocesser(
                resized_obj_img,
                resized_obj_mask,
                self.bg_img,
                self.bg_mask,
                Pixel(person_x, person_y),
                person_height,
                person_width,
                person_value,
            )
            self.images_to_save.append({"name": "unblended_image.png", "img":unblended_image})
            self.images_to_save.append({"name": "blended_image.png", "img": fg_bg_img})
            plt.imshow(fg_bg_img)
            plt.show()
        for image in self.images_to_save:
            save_dir = os.path.join(self.image_savedir, image["name"])
            img = cv2.cvtColor(image["img"], cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_dir, img)

if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
