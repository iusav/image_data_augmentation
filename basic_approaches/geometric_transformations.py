
# Transformation/ Translation

import cv2
import numpy as np
import random
from pymatting import *

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

    obj_contours, obj_hierarchy = cv2.findContours(obj_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    obj_areas = [cv2.contourArea(obj_contour) for obj_contour in obj_contours]

    # Ratio between object and background area
    # Min ratio for object chosing
    # default 0, i.e. chose objects with different size
    # can change obj_bg_ratio, for Example obj_bg_ratio = 0.005
    obj_bg_ratio = 0

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

# Random place finding
def random_place_finder(BGmask, ground_value, sidewalk_value, road_value, BGheight, BGwidth):
    gray_BGmask = cv2.cvtColor(BGmask, cv2.COLOR_RGB2GRAY)

    BGroad_thresh = np.where(
        (gray_BGmask == ground_value) | (gray_BGmask == sidewalk_value) | (gray_BGmask == road_value), gray_BGmask, 0)

    BGroad_contours, BGroad_hierarchy = cv2.findContours(BGroad_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = [cv2.contourArea(contour) for contour in BGroad_contours]

    area_id = contour_areas.index(max(contour_areas))

    bg_mask = np.full((BGheight, BGwidth), 0, np.uint8)
    bg_mask = cv2.drawContours(bg_mask, BGroad_contours, contourIdx=area_id, color=255, thickness=-1)

    place_koordinates = np.where(bg_mask == 255)
    size_road_value = len(place_koordinates[0])
    random_place = random.randrange(size_road_value)
    stand_y, stand_x = np.array([place_koordinates[0][random_place], place_koordinates[1][random_place]])

    return stand_y, stand_x

# Size of person finding
def person_size_finder(stand_y, w, h, obj_height, obj_width):
    M = np.array([[1000., 1.], [540., 1.]])
    v = np.array([800., 170.])
    solve = np.linalg.solve(M, v)

    stand_obj_height = round(solve[0] * stand_y + solve[1])
    stand_obj_width = round(w / h * stand_obj_height)

    obj_area = int(obj_height * obj_width * 1.2)
    stand_obj_area = int(stand_obj_height * stand_obj_width)

    if obj_area <= stand_obj_area:
        return False
    else:
        return stand_obj_height, stand_obj_width

# Matting function
def border_blender(img, mask):
    def blur_parameter_finder(mask):
        M_kern = np.array([[100000., 1.], [7000., 1.]])
        v_kern = np.array([11., 5.])
        solve_kern = np.linalg.solve(M_kern, v_kern)

        M_iter = np.array([[100000., 1.], [7000., 1.]])
        v_iter = np.array([2., 2.])
        solve_iter = np.linalg.solve(M_iter, v_iter)

        gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        mask_contours, mask_hierarchy = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        mask_area = cv2.contourArea(mask_contours[0])

        kern_value = round(solve_kern[0] * mask_area + solve_kern[1])
        iter_value = round(solve_iter[0] * mask_area + solve_iter[1])

        return kern_value, iter_value


    def trimap_creater(mask):
        kernel_value, iteration_value = blur_parameter_finder(mask)
        eros_iter_value = iteration_value
        dil_iter_value = iteration_value

        kernel = np.ones((kernel_value, kernel_value), np.uint8)

        copy_er_mask = mask.copy()
        mask_erosion = cv2.erode(copy_er_mask, kernel, iterations=eros_iter_value)
        mask_erosion = np.where(mask_erosion == 0, 0, 255).astype(np.uint8)

        copy_dil_mask = mask.copy()
        mask_dilation = cv2.dilate(copy_dil_mask, kernel, iterations=dil_iter_value)
        mask_dilation = np.where(mask_dilation == 0, 0, 128).astype(np.uint8)

        added_masks = cv2.add(mask_erosion, mask_dilation)

        return added_masks


    def matting(img, trimap):
        img = img/255.0
        trimap = cv2.cvtColor(trimap, cv2.COLOR_RGB2GRAY)
        trimap = trimap/255.0
        alpha = estimate_alpha_cf(img, trimap)

        # estimate foreground from image and alpha
        foreground = estimate_foreground_ml(img, alpha)

        return foreground, alpha

    def border_smoother(mask):
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return opening

    trimap_mask = trimap_creater(mask)

    new_img, alpha = matting(img, trimap_mask)
    new_mask = (alpha * 255).astype(np.uint8)
    new_mask = np.where(new_mask >= 110, 255, 0).astype(np.uint8)
    border_smoother_mask = border_smoother(new_mask)
    smoother_mask = np.full((border_smoother_mask.shape[0], border_smoother_mask.shape[1], 3), 0, np.uint8)
    smoother_mask[:, :, 0] = border_smoother_mask
    smoother_mask[:, :, 1] = border_smoother_mask
    smoother_mask[:, :, 2] = border_smoother_mask

    return new_img, new_mask, alpha, smoother_mask, trimap_mask

# Img and mask of object resizing
def obj_resizer(obj_img, obj_mask,
                stand_obj_height,
                stand_obj_width,
                person_value):
    resized_obj_img = cv2.resize(obj_img, (stand_obj_width, stand_obj_height), interpolation=cv2.INTER_CUBIC)

    binary_obj_mask = np.where(obj_mask==person_value, 255, 0).astype(np.uint8)
    resized_obj_mask = cv2.resize(binary_obj_mask, (stand_obj_width, stand_obj_height), interpolation=cv2.INTER_NEAREST)

    resized_obj_img, resized_obj_mask, alpha, smoother_mask, trimap_mask = border_blender(resized_obj_img,
                                                                             resized_obj_mask)

    return resized_obj_img, resized_obj_mask, alpha, smoother_mask, trimap_mask

# Foreground and background preprocessing
def fg_bg_preprocesser(resized_obj_img,
                       resized_obj_mask,
                       alpha,
                       background,
                       background_mask,
                       stand_y,
                       stand_x,
                       stand_obj_height,
                       stand_obj_width,
                       BGheight,
                       BGwidth,
                       person_value):
    fg_bg_mask = np.full((BGheight, BGwidth, 3), 0, np.uint8)
    obj_start_y = stand_y - stand_obj_height
    obj_start_x = stand_x - stand_obj_width // 2
    obj_end_y = stand_y
    obj_end_x = obj_start_x + stand_obj_width

    if obj_start_y < 0:
        resized_obj_img = resized_obj_img[-obj_start_y:, :]
        resized_obj_mask = resized_obj_mask[-obj_start_y:, :]
        alpha = alpha[-obj_start_y:, :]
        obj_start_y = 0

    if obj_start_x < 0:
        resized_obj_img = resized_obj_img[:, -obj_start_x:]
        resized_obj_mask = resized_obj_mask[:, -obj_start_x:]
        alpha = alpha[:, -obj_start_x:]
        obj_start_x = 0

    elif obj_end_x > BGwidth:
        resized_obj_img = resized_obj_img[:, :(BGwidth - obj_start_x)]
        resized_obj_mask = resized_obj_mask[:, :(BGwidth - obj_start_x)]
        alpha = alpha[:, :(BGwidth - obj_start_x)]
        obj_end_x = BGwidth

    fg_bg_img = background.copy()

    foreground = resized_obj_img; background = fg_bg_img[obj_start_y:obj_end_y, obj_start_x:obj_end_x]/255.
    part_fg_bg_img = blend(foreground, background, alpha)*255

    fg_bg_img[obj_start_y:obj_end_y, obj_start_x:obj_end_x] = part_fg_bg_img

    fg_bg_mask[obj_start_y:obj_end_y, obj_start_x:obj_end_x] = resized_obj_mask
    fg_bg_mask = np.where(fg_bg_mask == 255, person_value, background_mask).astype(np.uint8)

    return fg_bg_img, fg_bg_mask