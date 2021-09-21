# Transformation/ Translation

import cv2
import numpy as np
import random


# Data fliping
def data_fliper(img, mask):
    true_false_list = [True, False]
    flip_choice = random.choice(true_false_list)

    if flip_choice == True:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    return img, mask


# Object preprocessing
def obj_preprocesser(FGimg, FGmask, BGheight, BGwidth, person_value, FGheight, FGwidth):
    gray_FGmask = cv2.cvtColor(FGmask, cv2.COLOR_RGB2GRAY)
    obj_thresh = np.where(gray_FGmask == person_value, 255, 0).astype(np.uint8)

    contours, hierarchy = cv2.findContours(obj_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Finding the hierarchy for each object
    objHierarchy = []
    objContours = []
    row_hierarchy = None
    contour = None
    for i in range(hierarchy.shape[1]):
        if hierarchy[0][i][-1] == -1:
            if (row_hierarchy is not None):
                objHierarchy.append([row_hierarchy])
                objContours.append(contour)

            row_hierarchy = []
            contour = []

        row_hierarchy.append(hierarchy[0][i].tolist())
        contour.append(contours[i])

        if (row_hierarchy is not None and hierarchy.shape[1] == i + 1):
            objHierarchy.append([row_hierarchy])
            objContours.append(contour)

    # List of Areas
    obj_areas = [cv2.contourArea(objContour[0]) for objContour in objContours]

    # Ratio between object and background area
    # Min ratio for object chosing
    # default 0.01, i.e. obj_area / bg_area
    # can change obj_bg_ratio, for Example obj_bg_ratio = 0.005
    obj_bg_ratio = 0.01

    fg_area = int(FGheight * FGwidth)

    obj_areas_ratio = [obj_area / fg_area for obj_area in obj_areas]
    obj_areas_ratio = np.array(obj_areas_ratio)

    person_ids = np.where(obj_areas_ratio >= obj_bg_ratio)[0]
    if len(person_ids) == 0:
        raise IOError("didn't person find")
    else:
        person_id = np.random.choice(person_ids)
        obj_rect_x, obj_rect_y, obj_rect_w, obj_rect_h = cv2.boundingRect(objContours[person_id][0])

        obj_mask = np.full((FGheight, FGwidth), 0, np.uint8)
        obj_mask = cv2.drawContours(obj_mask, objContours[person_id], contourIdx=-1, color=person_value, thickness=-1)

        # croped object checking
        topX_count = np.count_nonzero(obj_mask[obj_rect_y, obj_rect_x:obj_rect_x + obj_rect_w - 1] == person_value)  # np.unique(obj_mask[0,:])

        downX_count = np.count_nonzero(
            obj_mask[obj_rect_y + obj_rect_h - 1, obj_rect_x:obj_rect_x + obj_rect_w - 1] == person_value)  # np.unique(obj_mask[obj_mask.shape[0]-1, :])

        leftY_count = np.count_nonzero(obj_mask[obj_rect_y:obj_rect_y + obj_rect_h - 1, obj_rect_x] == person_value)  # np.unique(obj_mask[:, 0])

        reightY_count = np.count_nonzero(
            obj_mask[obj_rect_y:obj_rect_y + obj_rect_h - 1, obj_rect_x + obj_rect_w - 1] == person_value)  # np.unique(obj_mask[:, obj_mask.shape[1]-1])

        crop_ratio = 1000
        crop_val = int(obj_areas[person_id] / crop_ratio)

        if (crop_val <= topX_count) or (crop_val <= downX_count) or (crop_val <= leftY_count) or (
                crop_val <= reightY_count):
            raise IOError("object is resized")
        else:
            obj_mask = cv2.cvtColor(obj_mask, cv2.COLOR_GRAY2RGB)
            obj_mask = obj_mask[obj_rect_y:obj_rect_y + obj_rect_h, obj_rect_x:obj_rect_x + obj_rect_w]

            obj_img = FGimg[obj_rect_y:obj_rect_y + obj_rect_h, obj_rect_x:obj_rect_x + obj_rect_w]

            return  obj_img, obj_mask, obj_rect_x, obj_rect_y, obj_rect_w, obj_rect_h

def test_for_occlusion(x, y, bg_mask, person_mask, occlusion_rate):
    obj_height = person_mask.shape[0]
    obj_width = person_mask.shape[1]
    obj_start_y = y - obj_height
    obj_start_x = x - obj_width // 2
    if obj_start_y < 0:
        obj_start_y = 0
    if obj_start_x < 0:
        obj_start_x = 0
    obj_end_y = obj_start_y + obj_height
    obj_end_x = obj_start_x + obj_width
    if obj_end_x > bg_mask.shape[1]:
        obj_end_x = bg_mask.shape[1]
        obj_start_x = obj_end_x - obj_width
    if obj_end_y > bg_mask.shape[0]:
        obj_end_y = bg_mask.shape[0]
        obj_start_y = obj_end_y - obj_height
    assert(obj_end_x-obj_start_x == obj_width)
    assert(obj_end_y-obj_start_y == obj_height)
    cimg = np.full((bg_mask.shape[0], bg_mask.shape[1]), 0, np.uint8)
    cimg[obj_start_y:obj_end_y, obj_start_x:obj_end_x] = person_mask
    occlusion_mask = cv2.bitwise_and(bg_mask, cimg)
    number_pixel_person = np.transpose(np.where(person_mask > 0)).shape[0]
    number_pixel_occlusion = np.transpose(np.where(occlusion_mask > 0)).shape[0]

    return number_pixel_occlusion / number_pixel_person > occlusion_rate


def suitable_place_finder(bg_mask, ground_value, sidewalk_value, road_value, obj_rect_y, obj_rect_h):
    gray_bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_RGB2GRAY)

    bg_height = gray_bg_mask.shape[0]
    bg_width = gray_bg_mask.shape[1]
    BGroad_thresh = np.where(
        (gray_bg_mask == ground_value)
        | (gray_bg_mask == sidewalk_value)
        | (gray_bg_mask == road_value),
        gray_bg_mask,
        0,
    )

    bg_road_contours, _ = cv2.findContours(
        BGroad_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_areas = [cv2.contourArea(contour) for contour in bg_road_contours]

    area_id = contour_areas.index(max(contour_areas))
    y_max = obj_rect_y + obj_rect_h
    bg_mask = np.full((bg_height, bg_width), 0, np.uint8)
    bg_mask = cv2.drawContours(
        bg_mask, bg_road_contours, contourIdx=area_id, color=255, thickness=-1
    )

    bg_mask[y_max:, :] = 0

    return np.where(bg_mask == 255)


def force_occlusion(
    bg_mask,
    person_mask,
    ground_value,
    sidewalk_value,
    road_value,
    obstacle_values,
    y,
    h,
    occlusion_rate,
):
    gray_bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_RGB2GRAY)
    bg_obstacle_tresh = np.full(
        (gray_bg_mask.shape[0], gray_bg_mask.shape[1]), 0, np.uint8
    )
    for obstacle_value in obstacle_values:
        bg_obstacle_tresh = bg_obstacle_tresh + np.where(
            gray_bg_mask == obstacle_value, gray_bg_mask, 0
        ).astype(np.uint8)
    _, person_mask_binary = cv2.threshold(
        cv2.cvtColor(person_mask, cv2.COLOR_RGB2GRAY), 1, 255, cv2.THRESH_BINARY
    )
    person_height = person_mask_binary.shape[0]
    person_width = person_mask_binary.shape[1]
    place_coordinates = np.transpose(
        suitable_place_finder(bg_mask, ground_value, sidewalk_value, road_value, y, h)
    )
    old_x = 10000
    old_y = 10000
    for x, y in place_coordinates:
        dx = x.item() - old_x
        dy = y.item() - old_y
        if abs(dx) < 2 * person_width and abs(dy) < person_height / 2:
            continue
        else:
            if test_for_occlusion(
                x, y, bg_obstacle_tresh, person_mask_binary, occlusion_rate
            ):
                return x, y
            else:
                old_x = x
                old_y = y
    raise IOError("No occlusions")


# Random place finading
def random_place_finder(bg_mask, ground_value, sidewalk_value, road_value, obj_rect_y, obj_rect_h):
    place_coordinates = suitable_place_finder(
        bg_mask, ground_value, sidewalk_value, road_value, obj_rect_y, obj_rect_h
    )
    size_road_value = len(place_coordinates[0])
    if size_road_value == 0:
        raise IOError("didn't road find")
    else:
        random_place = random.randrange(size_road_value)
        bottom_pixel_person = np.array(
            [place_coordinates[0][random_place], place_coordinates[1][random_place]]
        )
        return bottom_pixel_person


# Size of person finding
def person_size_finder(obj_pos_y, obj_width, obj_height):
    # To find the size of the person, we assume that
    # person has a certain size at two different
    # positions.Let's write the two equations and solve them
    # h = y*a + b
    # h - size of the person (px)
    # y - position (px)

    # 800 = 1000*a + b
    # 170 = 540*a + b

    # solve = [a, b]
    solve = [1.37, -569.57]

    new_obj_height = round(solve[0] * obj_pos_y + solve[1])
    new_obj_width = round(obj_width / obj_height * new_obj_height)

    return new_obj_height, new_obj_width


# Img and mask of object resizing
def obj_resizer(obj_img, obj_mask, stand_obj_height, stand_obj_width, person_value):
    resized_obj_img = cv2.resize(
        obj_img, (stand_obj_width, stand_obj_height), interpolation=cv2.INTER_CUBIC
    )

    binary_obj_mask = np.where(obj_mask == person_value, 255, 0).astype(np.uint8)
    resized_obj_mask = cv2.resize(
        binary_obj_mask,
        (stand_obj_width, stand_obj_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized_obj_img, resized_obj_mask


def get_obj_start_end(obj_bottom_x, obj_bottom_y, obj_width, obj_height):
    obj_start_y = obj_bottom_y - obj_height
    obj_start_x = obj_bottom_x - obj_width // 2
    obj_end_y = obj_bottom_y
    obj_end_x = obj_start_x + obj_width
    return obj_start_x, obj_start_y, obj_end_x, obj_end_y


def borderBlender(foreground, background, alpha):
    foreground = foreground / 255
    background = background / 255

    alpha = cv2.bitwise_not(alpha)
    alpha = alpha / 255

    return ((alpha * foreground + (1 - alpha) * background)*255).astype(np.uint8)


# Foreground and background preprocessing
def fg_bg_preprocesser(resized_obj_img,
                       resized_obj_mask,
                       background,
                       background_mask,
                       bottom_pixel_person_x,
                       bottom_pixel_person_y,
                       stand_obj_height,
                       stand_obj_width,
                       bg_height,
                       bg_width,
                       person_value
                       ):
    fg_bg_mask = np.full((bg_height, bg_width, 3), 0, np.uint8)
    bg_mask = np.full((bg_height, bg_width, 3), 0, np.uint8)
    obj_start_y = bottom_pixel_person_y - stand_obj_height
    obj_start_x = bottom_pixel_person_x - stand_obj_width // 2
    obj_end_y = bottom_pixel_person_y
    obj_end_x = obj_start_x + stand_obj_width

    if obj_start_y < 0:
        resized_obj_img = resized_obj_img[-obj_start_y:, :]
        resized_obj_mask = resized_obj_mask[-obj_start_y:, :]
        obj_start_y = 0

    if obj_start_x < 0:
        resized_obj_img = resized_obj_img[:, -obj_start_x:]
        resized_obj_mask = resized_obj_mask[:, -obj_start_x:]
        obj_start_x = 0

    elif obj_end_x > bg_width:
        resized_obj_img = resized_obj_img[:, :(bg_width - obj_start_x)]
        resized_obj_mask = resized_obj_mask[:, :(bg_width - obj_start_x)]
        obj_end_x = bg_width

    kernel = np.ones((3, 3), np.uint8)
    erode_obj_mask = cv2.erode(resized_obj_mask, kernel, iterations=1)

    ksize = (3, 3)
    blur_obj_mask = cv2.blur(erode_obj_mask, ksize)

    fg_bg_img = background.copy()

    fg_bg_img[obj_start_y:obj_end_y, obj_start_x:obj_end_x] = resized_obj_img
    fg_bg_mask[obj_start_y:obj_end_y, obj_start_x:obj_end_x] = blur_obj_mask

    bg_mask[obj_start_y:obj_end_y, obj_start_x:obj_end_x] = erode_obj_mask

    # combine foreground+background
    overlapImg = borderBlender(background, fg_bg_img, fg_bg_mask)
    overlapMask = np.where(bg_mask == 255, person_value, background_mask).astype(np.uint8)
    alphaMask = fg_bg_mask

    return overlapImg, overlapMask, alphaMask