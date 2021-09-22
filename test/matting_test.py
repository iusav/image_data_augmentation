#!/usr/bin/env python3
import os
import sys
import unittest
import cv2
from matplotlib import pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from basic_approaches.basic_img_augmentation import (
    pathReader,
    data_loader,
    obj_preprocesser,
    person_value,
    get_obj_start_end,
)
from basic_approaches.geometric_transformations import fg_bg_preprocesser, obj_resizer


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
        "img": os.path.join(bg_files_path, "aachen_000031_000019_leftImg8bit.png"),
        "mask": os.path.join(bg_files_path, "aachen_000031_000019_gtFine_labelIds.png"),
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

    def insert_object_to_background(self):
        mask_gray = cv2.cvtColor(self.obj_mask, cv2.COLOR_RGB2GRAY)
        _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)
        masked_obj = cv2.bitwise_and(self.obj_img, self.obj_img, mask=mask_binary)
        obj_start_x, obj_start_y, obj_end_x, obj_end_y = get_obj_start_end(
            self.x, self.y + self.h, self.w, self.h
        )
        bg = self.bg_img[obj_start_y:obj_end_y, obj_start_x:obj_end_x].copy()
        import copy

        image = copy.deepcopy(self.bg_img)
        loc = np.where(mask_binary != 0)
        bg[loc] = masked_obj[loc]
        image[obj_start_y:obj_end_y, obj_start_x:obj_end_x] = bg
        return image, masked_obj

    def test_(self):
        resized_obj_img, resized_obj_mask = obj_resizer(
            self.obj_img, self.obj_mask, self.h, self.w, person_value
        )
        augmented_image, _, _ = fg_bg_preprocesser(
            resized_obj_img,
            resized_obj_mask,
            self.bg_img,
            self.bg_mask,
            self.x,
            self.y + self.h,
            self.h,
            self.w,
            self.bg_height,
            self.bg_width,
            person_value,
        )
        clipped_augmented_image, clipped_obj = self.insert_object_to_background()

        fig_freq = plt.figure()
        gs_freq = fig_freq.add_gridspec(3, 3)
        axs = fig_freq.add_subplot(gs_freq[0, 0])
        axs.set_title("Original Image")
        axs.imshow(self.bg_img)
        axs.axis("off")
        axs = fig_freq.add_subplot(gs_freq[0, 1])
        axs.set_title("Matted Augmented Image")
        axs.imshow(augmented_image)
        axs.axis("off")
        axs = fig_freq.add_subplot(gs_freq[0, 2])
        axs.set_title("Clipped Augmented Image")
        axs.imshow(clipped_augmented_image)
        axs.axis("off")
        bg_img_gray = cv2.cvtColor(self.bg_img, cv2.COLOR_RGB2GRAY)
        orig_fourier_image = np.fft.fftshift(np.fft.fft2(bg_img_gray))
        axs = fig_freq.add_subplot(gs_freq[1, 0])
        axs.set_title("Original Fourier Image")
        axs.imshow(np.log(np.abs(orig_fourier_image)))
        axs.axis("off")
        augmented_image_gray = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2GRAY)
        augmented_fourier_image = np.fft.fftshift(np.fft.fft2(augmented_image_gray))
        axs = fig_freq.add_subplot(gs_freq[1, 1])
        axs.set_title("Augmented Fourier Image")
        axs.axis("off")
        axs.imshow(np.log(np.abs(augmented_fourier_image)))
        clipped_img_gray = cv2.cvtColor(clipped_augmented_image, cv2.COLOR_RGB2GRAY)
        clipped_fourier_image = np.fft.fftshift(np.fft.fft2(clipped_img_gray))
        axs = fig_freq.add_subplot(gs_freq[1, 2])
        axs.set_title("Clipped Fourier Image")
        axs.imshow(np.log(np.abs(clipped_fourier_image)))
        axs.axis("off")
        axs = fig_freq.add_subplot(gs_freq[2, 0])
        axs.set_title("Difference Original Fourier Image")
        axs.imshow(np.log(np.abs(orig_fourier_image - orig_fourier_image)))
        axs.axis("off")
        axs = fig_freq.add_subplot(gs_freq[2, 1])
        axs.set_title("Difference Augmented Fourier Image")
        axs.imshow(np.log(np.abs(orig_fourier_image - augmented_fourier_image)))
        axs.axis("off")
        axs = fig_freq.add_subplot(gs_freq[2, 2])
        axs.set_title("Difference Clipped Fourier Image")
        axs.imshow(np.log(np.abs(orig_fourier_image - clipped_fourier_image)))
        axs.axis("off")
        plt.show()
        plt.imsave("orig_image.png", self.bg_img)
        plt.imsave("orig_image_freq.png", np.log(np.abs(orig_fourier_image)))
        plt.imsave("blended_image.png", augmented_image)
        plt.imsave("blended_image_freq.png", np.log(np.abs(augmented_fourier_image)))
        plt.imsave(
            "blended_image_freq_diff.png",
            np.log(np.abs(augmented_fourier_image - orig_fourier_image)),
        )
        plt.imsave("clipped_image.png", clipped_augmented_image)
        plt.imsave("clipped_image_freq.png", np.log(np.abs(clipped_fourier_image)))
        plt.imsave(
            "clipped_image_freq_diff.png",
            np.log(np.abs(orig_fourier_image - clipped_fourier_image)),
        )


if __name__ == "__main__":
    unittest.main()
    print("Everything passed")
