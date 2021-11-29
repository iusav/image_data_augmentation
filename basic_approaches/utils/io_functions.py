import json
import glob
import os
import cv2
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.costum_exceptions import ShutdownException


def path_reader(path):
    check_for_file(path)
    # Read paths of a CSV file
    with open(path, newline="") as fg_bg_data:
        data = json.load(fg_bg_data)
    return data


def data_name(path):
    name = "_".join(path["mask"].split("/")[-1].split("_")[:-2])

    return name

def check_img_and_mask(img_paths):
    for _, value in img_paths.items():
        check_for_file(value)

def fg_data_loader(fg_path):
    check_img_and_mask(fg_path)
    # Foreground paths
    img_fg_path = fg_path["img"]
    mask_fg_path = fg_path["mask"]

    fg_img = cv2.imread(img_fg_path)
    fg_img = cv2.cvtColor(fg_img, cv2.COLOR_BGR2RGB)
    fg_mask = cv2.imread(mask_fg_path)
    fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2RGB)
    with open(fg_path["polygons"], "r") as polygons_settings:
        polygons_dict = json.load(polygons_settings)
    return fg_img, fg_mask, polygons_dict

def bg_data_loader(bg_path):
    check_img_and_mask(bg_path)
    # Background paths
    img_bg_path = bg_path["img"]
    mask_bg_path = bg_path["mask"]

    bg_img = cv2.imread(img_bg_path)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
    bg_mask = cv2.imread(mask_bg_path)
    bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_BGR2RGB)
    with open(bg_path["camera"], "r") as camera_settings:
        camera_dict = json.load(camera_settings)
    return bg_img, bg_mask, camera_dict


def data_saver(save_directory, data_name, img, mask, alpha_mask, id_data):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

    img_path = os.path.join(
        save_directory, "img", data_name + "_" + str(id_data) + ".png"
    )
    mask_path = os.path.join(
        save_directory, "mask", data_name + "_" + str(id_data) + ".png"
    )
    alpha_mask_path = os.path.join(
        save_directory,
        "gp-gan_predict",
        "alpha_mask",
        data_name + "_" + str(id_data) + ".png",
    )

    cv2.imwrite(img_path, img)
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(alpha_mask_path, alpha_mask)

    return current_id, img_path


def current_id(save_directory):
    path_list = glob.glob(os.path.join(save_directory, "mask", "*"))

    if not path_list:
        current_id = 1
    else:
        current_id = int(len(path_list) + 1)
    return current_id


def check_for_file(path):
    if not os.path.isfile(path):
        print(f"I can't find a file at {path}.")
        raise (ShutdownException)


def check_for_folder(folder):
    # check if folder structure exists
    folders = [
        folder,
        os.path.join(folder, "img"),
        os.path.join(folder, "mask"),
        os.path.join(folder, "gp-gan_predict", "alpha_mask"),
    ]
    for f in folders:
        try:
            os.makedirs(f)
            print(f"Created {f}")
        except FileExistsError:
            # folder already exists
            pass
