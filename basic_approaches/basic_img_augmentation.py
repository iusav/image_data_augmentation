import multiprocessing
import os
import sys
import argparse

import time
from reprint import output
import random
from multiprocessing import Lock, Process, Queue, current_process, Value
from ctypes import c_char_p
import queue
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.geometric_transformations import *
from utils.datastructures import Pixel
from utils.costum_exceptions import ShutdownException, FailedAugmentation
from utils.io_functions import (
    path_reader,
    data_name,
    data_loader,
    data_saver,
    current_id,
    check_for_file,
    check_for_folder
)

# Background
road_value = 7
ground_value = 6
sidewalk_value = 8

# Foreground
person_value = 24
aug_person_value=50

# Occlusion
obstacle_values = [13, 14, 15 ,17, 19, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 50]

class textcolor:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

class AugmentationWorkerManager(multiprocessing.Process):
    def __init__(self, num_workers, task_queue, fg_path_list, bg_path_list, save_directory):
        multiprocessing.Process.__init__(self)
        self.exit = multiprocessing.Event()
        self.workers = []
        self.task_queue = task_queue
        self.save_directory = save_directory
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=os.path.join(save_directory, 'augmentation.log'),
                    filemode='w')
        # set up workers.
        for i in range(num_workers):
            worker = AugmentationWorker(task_queue, fg_path_list, bg_path_list)
            self.workers.append(worker)
            worker.p.start()


    def run(self):
        # this process continously updates the print inside the console to show the current state for each worker
        with output(output_type="dict", interval=0) as output_lines:
            while not self.exit.is_set():
                output_lines["Images left on the queue: "] = self.task_queue.qsize()
                if len(self.workers) == 1:
                    self.exit.set()
                for i, worker in enumerate(self.workers, 0):
                    if not worker.state.value == b"Shutdown":
                        try:
                            output_lines[
                                "Worker {}".format(i)
                            ] = worker.state.value.decode()
                        except UnicodeDecodeError:
                            pass
                    else:
                        self.workers.remove(worker)
                time.sleep(0.5)


class AugmentationWorker:
    def __init__(self, task_queue, fg_path_list, bg_path_list):
        # start the image augmentation for each worker
        self.p = Process(
            target=self.augmentImage, args=(task_queue, fg_path_list, bg_path_list)
        )
        # set the state as multiprocessing value so the thread can change the value of the member variable
        self.state = Value(c_char_p, b"init")

    def augmentImageInner(self, fg_path_list, bg_path_list, task_id):
        # get a random foreground and background
        fg_path = random.choice(fg_path_list)
        bg_path = random.choice(bg_path_list)

        # Data name chosing
        fg_name, bg_name = data_name(fg_path, bg_path)

        # Data loading
        self.state.value = b"loading.."
        fg_img, fg_mask, bg_img, bg_mask, camera_dict = data_loader(fg_path, bg_path)
        bg_height = bg_mask.shape[0]
        bg_width = bg_mask.shape[1]

        # -------- Transformation/ Translation -------- #
        # Foreground fliping
        self.state.value = b"Flip..."
        flip_fg_img, flip_fg_mask = data_fliper(fg_img, fg_mask)

        # Background fliping
        flip_bg_img, flip_bg_mask = data_fliper(bg_img, bg_mask)
        self.state.value = b"Preprocess objects..."
        # Object preprocessing
        try:
            obj_img, obj_mask, obj_rect_x, obj_rect_y, obj_rect_w, obj_rect_h = obj_preprocesser(
                flip_fg_img,
                flip_fg_mask,
                person_value
            )
        except OSError as e:
            raise FailedAugmentation(e)
        self.state.value = b"Find random place..."
        # Random place finding
        try:
            if force_occlusion_flag:
                bottom_pixel_person = force_occlusion(
                    flip_bg_mask,
                    obj_mask,
                    ground_value,
                    sidewalk_value,
                    road_value,
                    obstacle_values,
                    obj_rect_y,
                    obj_rect_h,
                    min_occlusion_ratio,
                )
            else:
                bottom_pixel_person= random_place_finder(
                    flip_bg_mask, ground_value, sidewalk_value, road_value, obj_rect_y, obj_rect_h
                )
        except IOError:
            raise FailedAugmentation("Could not find any road to place the object on.")
        # Size of person finding
        obj_mask_height = obj_mask.shape[0]
        obj_mask_width = obj_mask.shape[1]
        person_height = round(person_height_calculation(camera_dict, bottom_pixel_person.x, bottom_pixel_person.y))
        person_width = round(obj_rect_w / obj_rect_h * person_height)
        if person_height < 0 or person_width < 0:
            raise FailedAugmentation()
        self.state.value = b"Blending.."
        # Img and mask of object resizing
        try:
            resized_obj_img, resized_obj_mask = obj_resizer(
                obj_img, obj_mask, person_height, person_width, person_value
            )
        except ValueError:
            raise FailedAugmentation("Trimap did not contain any values=0")
        # Foreground and background preprocessing
        fg_bg_img, fg_bg_mask, alpha_mask = fg_bg_preprocesser(
            resized_obj_img,
            resized_obj_mask,
            flip_bg_img,
            flip_bg_mask,
            bottom_pixel_person,
            person_height,
            person_width,
            bg_height,
            bg_width,
            aug_person_value,
        )
        self.state.value = b"Saving..."
        # Data saving
        _, img_path = data_saver(save_directory, bg_name, fg_bg_img, fg_bg_mask, alpha_mask, task_id)
        logging.debug(f"Saved file to {img_path} \n Params:\n Position X: {bottom_pixel_person.x} Y: {bottom_pixel_person.y}\n Object W: {person_width} H: {person_height}\n")

    def augmentImage(self, task_queue, fg_path_list, bg_path_list):
        while True:
            try:
                """
                    try to get task from the queue. get_nowait() function will
                    raise queue.Empty exception if the queue is empty.
                    queue(False) function would do the same task also.
                """
                # pull a task number from the queue and try to augment the image
                task = task_queue.get_nowait()
                self.augmentImageInner(fg_path_list, bg_path_list, task)
            except queue.Empty:
                # if the queue is empty this worker can shutdown
                self.state.value = bytes(
                    "{}Shutdown{}".format(textcolor.WARNING, textcolor.ENDC), "utf-8"
                )
                break
            except FailedAugmentation as e:
                # if the augmentation failed for some reason we need to put the task back on the queue
                task_queue.put_nowait(task)
            except ShutdownException as e:
                # if something goes wrong the process should be closed
                self.state.value = bytes("Shutdown", "utf-8")
                break
        return True

if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))
    DEFAULT_DATASET_SIZE = 50
    DEFAULT_OUTPUT_PATH = os.path.abspath(os.path.join(file_dir, "../created_dataset"))
    DEFAULT_FG_PATH = os.path.abspath(
        os.path.join(file_dir, "../basic_approaches/citysc_fgPaths.json")
    )
    DEFAULT_BG_PATH = os.path.abspath(
        os.path.join(file_dir, "../basic_approaches/citysc_bgPaths.json")
    )
    DEFAULT_NUMBER_PROCESSES = 2
    DEFAULT_OCCLUSION_FLAG = False
    DEFAULT_MIN_OCCLUSION_RATIO = 0.4

    parser = argparse.ArgumentParser(
        description="Create the augmented dataset for cityscapes."
    )
    parser.add_argument(
        "--dataset-size",
        dest="dataset_size",
        type=int,
        default=DEFAULT_DATASET_SIZE,
        help="Choose the size of the created dataset",
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        default=DEFAULT_OUTPUT_PATH,
        help="Choose where the created images are saved to.",
    )
    parser.add_argument(
        "--fg",
        dest="fg_paths",
        default=DEFAULT_FG_PATH,
        help="Select the json files which where created by the path_creating script.",
    )
    parser.add_argument(
        "--bg",
        dest="bg_paths",
        default=DEFAULT_BG_PATH,
        help="Select the json files which where created by the path_creating script.",
    )
    parser.add_argument(
        "--process",
        dest="num_processes",
        type=int,
        default=DEFAULT_NUMBER_PROCESSES,
        help="Select the number of processes.",
    )
    parser.add_argument(
        "--force_occlusion",
        dest="force_occlusion",
        default=DEFAULT_OCCLUSION_FLAG,
        type=bool,
        help="Forces occlusion of pedestrians with objects.",
    )
    parser.add_argument(
        "--min_occlusion_ratio",
        dest="min_occlusion_ratio",
        default=DEFAULT_MIN_OCCLUSION_RATIO,
        type=float,
        help="Set the occlusion ratio of the pedestrian.",
    )

    args = parser.parse_args()
    dataset_size = args.dataset_size

    fg_paths = args.fg_paths
    bg_paths = args.bg_paths
    save_directory = args.output_path
    num_processes = args.num_processes
    force_occlusion_flag = args.force_occlusion
    min_occlusion_ratio = args.min_occlusion_ratio

    check_for_folder(save_directory)
    fg_path_list = path_reader(fg_paths)
    bg_path_list = path_reader(bg_paths)

    start_time = time.time()

    id_data = current_id(save_directory)
    # id_data = 1
    if id_data >= dataset_size:
        print(
            "There are already enough images in the dataset. Either increase the dataset size or delete some images."
        )
        sys.exit(0)
    print(f"Start at image id {id_data}")
    task_queue = Queue()
    # put as many task on the queue as we want to have images in our dataset
    for i in range(dataset_size - id_data):
        task_queue.put(i)

    manager = AugmentationWorkerManager(
        num_processes, task_queue, fg_path_list, bg_path_list, save_directory)
    manager.start()
    while not task_queue.empty and len(manager.workers) > 1:
        print("Done!")
    sys.exit(0)
