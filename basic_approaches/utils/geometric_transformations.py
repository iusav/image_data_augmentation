import cv2
import numpy as np
import random
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.datastructures import Pixel, Rectangle

# Data fliping
def data_fliper(img, mask):
    true_false_list = [True, False]
    flip_choice = random.choice(true_false_list)

    if flip_choice == True:
        img = cv2.flip(img, 1)
        mask = cv2.flip(mask, 1)

    return img, mask


# Object preprocessing
def obj_preprocesser(fg_img, fg_mask, person_value):
    gray_fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_RGB2GRAY)
    obj_thresh = np.where(gray_fg_mask == person_value, 255, 0).astype(np.uint8)

    contours, hierarchy = cv2.findContours(
        obj_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Finding the hierarchy for each object
    objHierarchy = []
    objContours = []
    row_hierarchy = None
    contour = None
    if not contours:
       raise IOError("Didn't find a suiting person in the image.")
    for i in range(hierarchy.shape[1]):
        if hierarchy[0][i][-1] == -1:
            if row_hierarchy is not None:
                objHierarchy.append([row_hierarchy])
                objContours.append(contour)

            row_hierarchy = []
            contour = []

        row_hierarchy.append(hierarchy[0][i].tolist())
        contour.append(contours[i])

        if row_hierarchy is not None and hierarchy.shape[1] == i + 1:
            objHierarchy.append([row_hierarchy])
            objContours.append(contour)

    # List of Areas
    obj_areas = [cv2.contourArea(objContour[0]) for objContour in objContours]

    # Ratio between object and background area
    # Min ratio for object chosing
    # default 0.01, i.e. obj_area / bg_area
    # can change obj_bg_ratio, for Example obj_bg_ratio = 0.005
    obj_bg_ratio = 0.01
    fg_height = fg_mask.shape[0]
    fg_width = fg_mask.shape[1]
    fg_area = int(fg_height * fg_width)

    obj_areas_ratio = [obj_area / fg_area for obj_area in obj_areas]
    obj_areas_ratio = np.array(obj_areas_ratio)

    person_ids = np.where(obj_areas_ratio >= obj_bg_ratio)[0]
    if len(person_ids) == 0:
        raise IOError("Didn't find a suiting person in the image.")
    else:
        person_id = np.random.choice(person_ids)
        obj_rect = Rectangle(*cv2.boundingRect(objContours[person_id][0]))

        obj_mask = np.full((fg_height, fg_width), 0, np.uint8)
        obj_mask = cv2.drawContours(
            obj_mask,
            objContours[person_id],
            contourIdx=-1,
            color=person_value,
            thickness=-1,
        )

        # croped object checking
        topX_count = np.count_nonzero(
            obj_mask[obj_rect.y, obj_rect.x : obj_rect.x + obj_rect.w - 1]
            == person_value
        )  # np.unique(obj_mask[0,:])

        downX_count = np.count_nonzero(
            obj_mask[
                obj_rect.y + obj_rect.h - 1, obj_rect.x : obj_rect.x + obj_rect.w - 1
            ]
            == person_value
        )  # np.unique(obj_mask[obj_mask.shape[0]-1, :])

        leftY_count = np.count_nonzero(
            obj_mask[obj_rect.y : obj_rect.y + obj_rect.h - 1, obj_rect.x]
            == person_value
        )  # np.unique(obj_mask[:, 0])

        reightY_count = np.count_nonzero(
            obj_mask[
                obj_rect.y : obj_rect.y + obj_rect.h - 1, obj_rect.x + obj_rect.w - 1
            ]
            == person_value
        )  # np.unique(obj_mask[:, obj_mask.shape[1]-1])

        crop_ratio = 1000
        crop_val = int(obj_areas[person_id] / crop_ratio)

        if (
            (crop_val <= topX_count)
            or (crop_val <= downX_count)
            or (crop_val <= leftY_count)
            or (crop_val <= reightY_count)
        ):
            raise IOError("object is resized")
        else:
            obj_mask = cv2.cvtColor(obj_mask, cv2.COLOR_GRAY2RGB)
            obj_mask = obj_mask[
                obj_rect.y : obj_rect.y + obj_rect.h,
                obj_rect.x : obj_rect.x + obj_rect.w,
            ]

            obj_img = fg_img[
                obj_rect.y : obj_rect.y + obj_rect.h,
                obj_rect.x : obj_rect.x + obj_rect.w,
            ]

            return obj_img, obj_mask, obj_rect


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
    assert obj_end_x - obj_start_x == obj_width
    assert obj_end_y - obj_start_y == obj_height
    cimg = np.full((bg_mask.shape[0], bg_mask.shape[1]), 0, np.uint8)
    cimg[obj_start_y:obj_end_y, obj_start_x:obj_end_x] = person_mask
    occlusion_mask = cv2.bitwise_and(bg_mask, cimg)
    number_pixel_person = np.transpose(np.where(person_mask > 0)).shape[0]
    number_pixel_occlusion = np.transpose(np.where(occlusion_mask > 0)).shape[0]

    return number_pixel_occlusion / number_pixel_person > occlusion_rate


def suitable_place_finder(bg_mask, ground_values, obj_rect_y, obj_rect_h):
    gray_bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_RGB2GRAY)

    bg_height = gray_bg_mask.shape[0]
    bg_width = gray_bg_mask.shape[1]
    bg_ground_thresh = np.full((bg_height, bg_width), 0, np.uint8)
    for ground_value in ground_values:
        bg_ground_thresh = bg_ground_thresh + np.where(
            gray_bg_mask == ground_value, gray_bg_mask, 0
        ).astype(np.uint8)

    bg_road_contours, _ = cv2.findContours(
        bg_ground_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
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
    bg_mask, person_mask, ground_values, obstacle_values, y, h, occlusion_rate
):
    gray_bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_RGB2GRAY)
    bg_obstacle_thresh = np.full(
        (gray_bg_mask.shape[0], gray_bg_mask.shape[1]), 0, np.uint8
    )
    for obstacle_value in obstacle_values:
        bg_obstacle_thresh = bg_obstacle_thresh + np.where(
            gray_bg_mask == obstacle_value, gray_bg_mask, 0
        ).astype(np.uint8)
    _, person_mask_binary = cv2.threshold(
        cv2.cvtColor(person_mask, cv2.COLOR_RGB2GRAY), 1, 255, cv2.THRESH_BINARY
    )
    person_height = person_mask_binary.shape[0]
    person_width = person_mask_binary.shape[1]
    place_coordinates = np.transpose(
        suitable_place_finder(bg_mask, ground_values, y, h)
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
                x, y, bg_obstacle_thresh, person_mask_binary, occlusion_rate
            ):
                return Pixel(x, y)
            else:
                old_x = x
                old_y = y
    raise IOError("No occlusions")


# Random place finading
def random_place_finder(bg_mask, ground_values, obj_rect_y, obj_rect_h):
    place_coordinates = suitable_place_finder(
        bg_mask, ground_values, obj_rect_y, obj_rect_h
    )
    size_road_value = len(place_coordinates[0])
    if size_road_value == 0:
        raise IOError("didn't road find")
    else:
        random_place = random.randrange(size_road_value)
        return Pixel(
            place_coordinates[1][random_place], place_coordinates[0][random_place]
        )


def get_rotation_matrix(params):
    pitch = params["extrinsic"]["pitch"]
    yaw = params["extrinsic"]["yaw"]
    rol = params["extrinsic"]["roll"]
    x11 = math.cos(yaw) * math.cos(pitch)
    x12 = math.cos(yaw) * math.sin(pitch) * math.sin(rol) - math.sin(yaw) * math.cos(
        rol
    )
    x13 = math.cos(yaw) * math.sin(pitch) * math.cos(rol) + math.sin(yaw) * math.sin(
        rol
    )
    x21 = math.sin(yaw) * math.cos(pitch)
    x22 = math.sin(yaw) * math.sin(pitch) * math.sin(rol) + math.cos(yaw) * math.cos(
        rol
    )
    x23 = math.sin(yaw) * math.sin(pitch) * math.cos(rol) - math.cos(yaw) * math.sin(
        rol
    )
    x31 = -math.sin(pitch)
    x32 = math.cos(pitch) * math.sin(rol)
    x33 = math.cos(pitch) * math.cos(rol)
    return np.array([[x11, x12, x13], [x21, x22, x23], [x31, x32, x33]])


def load_camera_params(params):
    x = params["extrinsic"]["x"]
    y = params["extrinsic"]["y"]
    z = params["extrinsic"]["z"]

    fx = params["intrinsic"]["fx"]
    fy = params["intrinsic"]["fy"]
    u0 = params["intrinsic"]["u0"]
    v0 = params["intrinsic"]["v0"]
    return x, y, z, fx, fy, u0, v0


def person_height_calculation(params, position_x, position_y):
    # Parameters
    x, y, z, fx, fy, u0, v0 = load_camera_params(params)

    # pixel
    q = np.array([[position_x], [position_y]])

    # Translation vector
    t = np.array([[x], [y], [z]])

    # Rotation matrix
    r = get_rotation_matrix(params)

    r1 = np.transpose(r)
    t = -np.dot(r1, t)

    # Intrinsic parameters matrix

    k = np.array([[fx, 0, u0], [0, fy, v0], [0, 0, 1]])
    i = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    c = np.dot(k, i)

    pg = transform_point_2D_to_3D(q, r1, t, c, 0)
    x0 = pg[0, 0, 0]
    y0 = pg[0, 0, 1]
    # height of a person
    z0 = 1.77
    pg1 = np.array([[x0], [y0], [z0]])
    pg3 = transform_point_3D_to_2D(pg1, r1, t, c)
    z = int(pg3[1, 0])
    height = position_y - z

    return height


# Pixel to world
def transform_point_2D_to_3D(point2D, rVec, tVec, cameraMat, height):
    """
    Function used to convert given 2D points back to real-world 3D points
    point2D  : An array of 2D points
    rVec     : Rotation vector
    tVec     : Translation vector
    cameraMat: Camera Matrix used in solvePnP
    height   : Height in real-world 3D space
    Return   : output_array: Output array of 3D points

    """
    point3D = []
    point2D = (np.array(point2D, dtype="float32")).reshape(-1, 2)
    numPts = point2D.shape[0]
    point2D_op = np.hstack((point2D, np.ones((numPts, 1))))
    rMat = rVec
    rMat_inv = np.linalg.inv(rMat)
    kMat_inv = np.linalg.inv(cameraMat)
    for point in range(numPts):
        uvPoint = point2D_op[point, :].reshape(3, 1)
        tempMat = np.matmul(rMat_inv, kMat_inv)
        tempMat1 = np.matmul(tempMat, uvPoint)
        tempMat2 = np.matmul(rMat_inv, tVec)
        s = (height + tempMat2[2]) / tempMat1[2]
        p = tempMat1 * s - tempMat2
        point3D.append(p)

    point3D = (np.array(point3D, dtype="float32")).reshape([-1, 1, 3])
    return point3D


# World to pixel
def transform_point_3D_to_2D(point2D, rv, tv, c):
    """
    Function used to convert given 3D points back to 2D points
    point2D  : An array of 3D points
    rVec     : Rotation vector
    tVec     : Translation vector
    cameraMat: Camera Matrix used in solvePnP
    Return   : output_array: Output array of 2D points

    """
    rr = np.array(
        [
            [rv[0, 0], rv[0, 1], rv[0, 2], tv[0, 0]],
            [rv[1, 0], rv[1, 1], rv[1, 2], tv[1, 0]],
            [rv[2, 0], rv[2, 1], rv[2, 2], tv[2, 0]],
        ]
    )
    pg2 = np.array([[point2D[0, 0]], [point2D[1, 0]], [point2D[2, 0]], [1]])
    a1 = np.dot(c, rr)
    a2 = np.dot(a1, pg2)
    pixel = a2 / a2[2, 0]
    return pixel


# Img and mask of object resizing
def obj_resizer(obj_img, obj_mask, obj_height, obj_width, person_value):
    resized_obj_img = cv2.resize(
        obj_img, (obj_width, obj_height), interpolation=cv2.INTER_CUBIC
    )

    binary_obj_mask = np.where(obj_mask == person_value, 255, 0).astype(np.uint8)
    resized_obj_mask = cv2.resize(
        binary_obj_mask, (obj_width, obj_height), interpolation=cv2.INTER_NEAREST
    )
    return resized_obj_img, resized_obj_mask


def get_obj_start_end(obj_bottom_x, obj_bottom_y, obj_width, obj_height):
    obj_start_y = obj_bottom_y - obj_height
    obj_start_x = obj_bottom_x - obj_width // 2
    obj_end_y = obj_bottom_y
    obj_end_x = obj_start_x + obj_width
    return obj_start_x, obj_start_y, obj_end_x, obj_end_y


def border_blender(foreground, background, alpha):
    foreground = foreground / 255
    background = background / 255

    alpha = cv2.bitwise_not(alpha)
    alpha = alpha / 255

    return ((alpha * foreground + (1 - alpha) * background) * 255).astype(np.uint8)


# Foreground and background preprocessing
def fg_bg_preprocesser(
    resized_obj_img,
    resized_obj_mask,
    background,
    background_mask,
    bottom_pixel_person,
    obj_height,
    obj_width,
    person_value,
):
    bg_height = background_mask.shape[0]
    bg_width = background_mask.shape[1]
    fg_bg_mask = np.full((bg_height, bg_width, 3), 0, np.uint8)
    bg_mask = np.full((bg_height, bg_width, 3), 0, np.uint8)
    obj_start_x, obj_start_y, obj_end_x, obj_end_y = get_obj_start_end(
        bottom_pixel_person.x, bottom_pixel_person.y, obj_width, obj_height
    )
    if obj_start_y < 0:
        resized_obj_img = resized_obj_img[-obj_start_y:, :]
        resized_obj_mask = resized_obj_mask[-obj_start_y:, :]
        obj_start_y = 0

    if obj_start_x < 0:
        resized_obj_img = resized_obj_img[:, -obj_start_x:]
        resized_obj_mask = resized_obj_mask[:, -obj_start_x:]
        obj_start_x = 0

    elif obj_end_x > bg_width:
        resized_obj_img = resized_obj_img[:, : (bg_width - obj_start_x)]
        resized_obj_mask = resized_obj_mask[:, : (bg_width - obj_start_x)]
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
    overlapImg = border_blender(background, fg_bg_img, fg_bg_mask)
    overlapMask = np.where(bg_mask == 255, person_value, background_mask).astype(
        np.uint8
    )
    alphaMask = fg_bg_mask

    return overlapImg, overlapMask, alphaMask
