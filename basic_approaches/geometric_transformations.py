
# Transformation/ Translation

import cv2
import numpy as np
import random

def obj_preprocesser(FGimg, FGmask, BGheight, BGwidth, person_value, FGheight, FGwidth):
    gray_FGmask = cv2.cvtColor(FGmask, cv2.COLOR_RGB2GRAY)
    obj_thresh = np.where(gray_FGmask == person_value, 255, 0).astype(np.uint8)

    obj_contours, obj_hierarchy = cv2.findContours(obj_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    obj_areas = [cv2.contourArea(obj_contour) for obj_contour in obj_contours]

    obj_bg_ratio = 0.0143
    bg_area = int(BGheight * BGwidth)

    obj_areas_ratio = [obj_area / bg_area for obj_area in obj_areas]
    obj_areas_ratio = np.array(obj_areas_ratio)
    person_ids = np.where(obj_areas_ratio > obj_bg_ratio)[0]
    if len(person_ids) == 0:
        return False
    else:
        person_id = np.random.choice(person_ids)

        x, y, w, h = cv2.boundingRect(obj_contours[person_id])

        obj_mask = np.full((FGheight, FGwidth), 0, np.uint8)
        obj_mask = cv2.drawContours(obj_mask, obj_contours, contourIdx=person_id, color=person_value, thickness=-1)
        obj_mask = cv2.cvtColor(obj_mask, cv2.COLOR_GRAY2RGB)
        obj_mask = obj_mask[y:y + h, x:x + w]

        obj_img = FGimg[y:y + h, x:x + w]

        return obj_img, obj_mask, x, y, w, h


def random_place_finder(augment_BGmask, ground_value, sidewalk_value, road_value, BGheight, BGwidth):
    gray_BGmask = cv2.cvtColor(augment_BGmask, cv2.COLOR_RGB2GRAY)

    BGroad_thresh = np.where(
        (gray_BGmask == ground_value) | (gray_BGmask == sidewalk_value) | (gray_BGmask == road_value), gray_BGmask, 0)

    BGroad_contours, BGroad_hierarchy = cv2.findContours(BGroad_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = [cv2.contourArea(contour) for contour in BGroad_contours]
    max_area_id = contour_areas.index(max(contour_areas))
    max_BGroad_contour = BGroad_contours[max_area_id]

    BGroad_min_y = min([coordinates[0][1] for coordinates in max_BGroad_contour])

    bg_mask = np.full((BGheight, BGwidth), 0, np.uint8)
    bg_mask = cv2.drawContours(bg_mask, BGroad_contours, contourIdx=max_area_id, color=255, thickness=-1)

    place_koordinates = np.where(bg_mask == 255)
    size_road_value = len(place_koordinates[0])
    random_place = random.randrange(size_road_value)
    stand_y, stand_x = np.array([place_koordinates[0][random_place], place_koordinates[1][random_place]])

    return stand_y, stand_x, BGroad_min_y


def person_size_finder(stand_y, w, h):
    M1 = np.array([[1000., 1.], [540., 1.]])
    v1 = np.array([800., 170.])
    solve = np.linalg.solve(M1, v1)

    stand_obj_height = round(solve[0] * stand_y + solve[1])
    stand_obj_width = round(w / h * stand_obj_height)

    return stand_obj_height, stand_obj_width

def obj_resizer(obj_img, obj_mask, stand_obj_height, stand_obj_width, person_value):
    resized_obj_img = cv2.resize(obj_img, (stand_obj_width, stand_obj_height), interpolation=cv2.INTER_CUBIC)

    binary_obj_mask = np.where(obj_mask==person_value, 255, 0).astype(np.uint8)
    resized_obj_mask = cv2.resize(binary_obj_mask, (stand_obj_width, stand_obj_height), interpolation=cv2.INTER_NEAREST)

    return resized_obj_img, resized_obj_mask


def fg_bg_preprocesser(resized_obj_img, resized_obj_mask, background, stand_y, stand_x, stand_obj_height, stand_obj_width, BGheight, BGwidth):
    fg_bg_mask = np.full((BGheight, BGwidth, 3), 0, np.uint8)
    obj_start_y = stand_y - stand_obj_height
    obj_start_x = stand_x - stand_obj_width // 2
    obj_end_y = stand_y
    obj_end_x = obj_start_x + stand_obj_width

    if obj_start_y < 0:
        resized_obj_img = resized_obj_img[-obj_start_y:, :]
        resized_obj_mask = resized_obj_mask[-obj_start_y:, :]
        obj_start_y = 0

    if obj_start_x < 0:
        resized_obj_img = resized_obj_img[:, -obj_start_x:]
        resized_obj_mask = resized_obj_mask[:, -obj_start_x:]
        obj_start_x = 0

    elif obj_end_x > BGwidth:
        resized_obj_img = resized_obj_img[:, :(BGwidth - obj_start_x)]
        resized_obj_mask = resized_obj_mask[:, :(BGwidth - obj_start_x)]
        obj_end_x = BGwidth

    fg_bg_img = background.copy()

    fg_bg_img[obj_start_y:obj_end_y, obj_start_x:obj_end_x] = resized_obj_img
    fg_bg_mask[obj_start_y:obj_end_y, obj_start_x:obj_end_x] = resized_obj_mask

    return fg_bg_img, fg_bg_mask


def data_fliper(img, mask):
    true_false_list = [True, False]
    flip_choice = random.choice(true_false_list)

    if flip_choice == True:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    return img, mask