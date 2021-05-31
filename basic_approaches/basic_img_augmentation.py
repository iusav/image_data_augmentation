import os
import sys
import cv2
import csv
import argparse
import glob
import time
import json
from reprint import output
import random
import numpy as np
import matplotlib.pyplot as plt
from geometric_transformations import *
from multiprocessing import Lock, Process, Queue, current_process, Value
from ctypes import c_char_p
import queue

# Background
road_value = 7
ground_value = 6
sidewalk_value = 8

# Foreground
person_value = 24

# Occlusion
obstacle_values = [13, 14, 15 ,17, 19, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

class textcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class FailedAugmentation(Exception):
    pass

class AugmentationWorkerManager:
    def __init__(self, num_workers, task_queue, fg_path_list, bg_path_list):
        self.workers = []
        self.task_queue = task_queue
        # set up workers. Is num_workers-1 because one thread is reserved for the worker manager
        for i in range(num_workers-1):
            worker = AugmentationWorker(task_queue, fg_path_list, bg_path_list)
            self.workers.append(worker)
            worker.p.start()
        manager = Process(target=self.monitorProcesses)
        manager.start()

        for worker in self.workers:
            worker.p.join()

    def monitorProcesses(self):
        # this process continously updates the print inside the console to show the current state for each worker
        with output(output_type="dict", interval=0) as output_lines:
            while True:
                output_lines["Images left on the queue: "] = self.task_queue.qsize()
                for i, worker in enumerate(self.workers,1):
                    output_lines["Worker {}".format(i)] = worker.state.value.decode()
                time.sleep(0.1)

class AugmentationWorker:
    def __init__(self, task_queue, fg_path_list, bg_path_list):
        # start the image augmentation for each worker
        self.p = Process(target=self.augmentImage, args=(task_queue, fg_path_list, bg_path_list))
        # set the state as multiprocessing value so the thread can change the value of the member variable
        self.state = Value(c_char_p, b"init")
    def augmentImageInner(self, fg_path_list, bg_path_list, task_id):
        # get a random foreground and background
        fg_path = random.choice(fg_path_list)
        bg_path = random.choice(bg_path_list)

        # Data name chosing
        FGname, BGname = data_name(fg_path, bg_path)

        # Data loading
        self.state.value = b"loading.."
        FGimg, FGmask, BGimg, BGmask, camera_dict = data_loader(fg_path, bg_path)
        FGheight = FGmask.shape[0]; FGwidth = FGmask.shape[1]
        BGheight = BGmask.shape[0]; BGwidth = BGmask.shape[1]


        # -------- Transformation/ Translation -------- #
        # Foreground fliping
        self.state.value = b"Flip..."
        flip_FGimg, flip_FGmask = data_fliper(FGimg, FGmask)

        # Background fliping
        flip_BGimg, flip_BGmask = data_fliper(BGimg, BGmask)
        self.state.value = b"Preprocess objects..."
        # Object preprocessing
        try:
            obj_img, obj_mask, x,y,w,h = obj_preprocesser(flip_FGimg,
                                                        flip_FGmask,
                                                        BGheight,
                                                        BGwidth,
                                                        person_value,
                                                        FGheight,
                                                        FGwidth)
        except OSError:
            raise FailedAugmentation()
        self.state.value = b"Find random place..."
        # Random place finding
        try:
            if force_occlusion_flag:
                bottom_pixel_person = force_occlusion(flip_BGmask, obj_mask, ground_value, sidewalk_value, road_value, obstacle_values, y, h, min_occlusion_ratio)
            else:
                bottom_pixel_person = random_place_finder(flip_BGmask,
                                                        ground_value,
                                                        sidewalk_value,
                                                        road_value,
                                                        y,
                                                        h)
        except IOError:
            raise FailedAugmentation("Could not find any road to place the object on.")
        # Size of person finding
        obj_mask_height = obj_mask.shape[0]; obj_mask_width = obj_mask.shape[1]
        stand_obj_height, stand_obj_width = person_size_finder(bottom_pixel_person[0],
                                                                w,
                                                                h,
                                                                obj_mask_height,
                                                                obj_mask_width)
        self.state.value = b"Matting.."
        # Img and mask of object resizing
        # Matting function using
        resized_obj_img, resized_obj_mask, alpha, smoother_mask, trimap_mask = obj_resizer(obj_img,
                                                                                            obj_mask,
                                                                                            stand_obj_height,
                                                                                            stand_obj_width,
                                                                                            person_value)

        # Foreground and background preprocessing
        fg_bg_img, fg_bg_mask = fg_bg_preprocesser(resized_obj_img,
                                                    smoother_mask,
                                                    alpha,
                                                    flip_BGimg,
                                                    flip_BGmask,
                                                    bottom_pixel_person[0], bottom_pixel_person[1],
                                                    stand_obj_height,
                                                    stand_obj_width,
                                                    BGheight,
                                                    BGwidth,
                                                    person_value)
        self.state.value = b"Saving..."
        # Data saving
        data_saver(BGname, fg_bg_img, fg_bg_mask, task_id)

    def augmentImage(self, task_queue, fg_path_list, bg_path_list):
        while True:
            try:
                '''
                    try to get task from the queue. get_nowait() function will
                    raise queue.Empty exception if the queue is empty.
                    queue(False) function would do the same task also.
                '''
                # pull a task number from the queue and try to augment the image
                task = task_queue.get_nowait()
                self.augmentImageInner(fg_path_list, bg_path_list, task)
            except queue.Empty:
                # if the queue is empty this worker can shutdowm
                self.state.value = bytes("{}Shutdown{}".format(textcolor.WARNING, textcolor.ENDC), "utf-8")
                break
            except FailedAugmentation as e:
                # if the augmentation failed for some reason we need to put the task back on the queue
                task_queue.put_nowait(task)
        return True

def pathReader(path):
    # Read paths of a CSV file
    with open(path, newline='') as fg_bg_data:
        data = json.load(fg_bg_data)
    return data

def data_name(fg_path,bg_path):
    FGname = '_'.join(fg_path["mask"].split('/')[-1].split('_')[:-2])
    BGname = '_'.join(bg_path["mask"].split('/')[-1].split('_')[:-2])

    return FGname, BGname

def data_loader(fg_path,bg_path):
    # Foreground paths
    imgFG_path = fg_path["img"]; maskFG_path = fg_path["mask"]

    # Background paths
    imgBG_path = bg_path["img"]; maskBG_path = bg_path["mask"]


    FGimg = cv2.imread(imgFG_path)
    FGimg = cv2.cvtColor(FGimg, cv2.COLOR_BGR2RGB)
    FGmask = cv2.imread(maskFG_path)
    FGmask = cv2.cvtColor(FGmask, cv2.COLOR_BGR2RGB)

    BGimg = cv2.imread(imgBG_path)
    BGimg = cv2.cvtColor(BGimg, cv2.COLOR_BGR2RGB)
    BGmask = cv2.imread(maskBG_path)
    BGmask = cv2.cvtColor(BGmask, cv2.COLOR_BGR2RGB)
    with open(bg_path["camera"], "r") as camera_settings:
        camera_dict = json.load(camera_settings)
    return FGimg, FGmask, BGimg, BGmask, camera_dict

def data_saver(data_name, img, mask, id_data):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

    img_path = os.path.join(save_directory, 'img', data_name + '_' + str(id_data) + '.jpg')
    mask_path = os.path.join(save_directory, 'mask', data_name + '_' + str(id_data) +'.png')

    cv2.imwrite(img_path, img)
    cv2.imwrite(mask_path, mask)

    return current_id

def current_id():
    path_list = glob.glob(os.path.join(save_directory, 'mask','*'))

    if not path_list:
        current_id = 1
    else:
        current_id = int(len(path_list)+1)
    return current_id

def checkForFolder(folder):
    # check if folder structure exists
    folders = [folder, os.path.join(folder, 'img'), os.path.join(folder, 'mask')]
    for f in folders:
        try:
            os.makedirs(f)
            print(f"Created {f}")
        except FileExistsError:
            # folder already exists
            pass

if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.realpath(__file__))
    DEFAULT_DATASET_SIZE = 50
    DEFAULT_OUTPUT_PATH = os.path.abspath(os.path.join(file_dir, "../created_dataset"))
    DEFAULT_FG_PATH = os.path.abspath(os.path.join(file_dir, "../basic_approaches/citysc_fgPaths.json"))
    DEFAULT_BG_PATH = os.path.abspath(os.path.join(file_dir, "../basic_approaches/citysc_bgPaths.json"))
    DEFAULT_NUMBER_PROCESSES = 4
    DEFAULT_OCCLUSION_FLAG = False
    DEFAULT_MIN_OCCLUSION_RATIO = 0.4

    parser = argparse.ArgumentParser(description='Create the augmented dataset for cityscapes.')
    parser.add_argument('--dataset-size', dest='dataset_size', type=int, default=DEFAULT_DATASET_SIZE,
                        help='Choose the size of the created dataset')
    parser.add_argument('--output-path', dest='output_path', default=DEFAULT_OUTPUT_PATH, help='Choose where the created images are saved to.')
    parser.add_argument('--fg', dest='fgPaths', default=DEFAULT_FG_PATH, help='Select the json files which where created by the path_creating script.')
    parser.add_argument('--bg', dest='bgPaths', default=DEFAULT_BG_PATH, help='Select the json files which where created by the path_creating script.')
    parser.add_argument('--process', dest='num_processes', default=DEFAULT_NUMBER_PROCESSES, help='Select the number of processes.')
    parser.add_argument('--force_occlusion', dest='force_occlusion', default=DEFAULT_OCCLUSION_FLAG, type=bool, help='Forces occlusion of pedestrians with objects.')
    parser.add_argument('--min_occlusion_ratio', dest='min_occlusion_ratio', default=DEFAULT_MIN_OCCLUSION_RATIO, type=float, help='Set the occlusion ratio of the pedestrian.')



    args = parser.parse_args()
    dataset_size = args.dataset_size

    fgPaths = args.fgPaths
    bgPaths = args.bgPaths
    save_directory = args.output_path
    num_processes = args.num_processes
    force_occlusion_flag = args.force_occlusion
    min_occlusion_ratio = args.min_occlusion_ratio

    checkForFolder(save_directory)
    fg_path_list = pathReader(fgPaths)
    bg_path_list = pathReader(bgPaths)

    start_time = time.time()

    id_data = current_id()
    #id_data = 1
    if id_data >= dataset_size:
        print("There are already enough images in the dataset. Either increase the dataset size or delete some images.")
        sys.exit(0)
    print(f"Start at image id {id_data}")
    task_queue = Queue()
    # put as many task on the queue as we want to have images in our dataset
    for i in range(dataset_size-id_data):
        task_queue.put(i)

    manager = AugmentationWorkerManager(num_processes, task_queue, fg_path_list, bg_path_list)
    while not task_queue.empty:
        pass
    print("----------------- %s seconds ----------------" % ( round((time.time() - start_time), 2) ))
    sys.exit(0)