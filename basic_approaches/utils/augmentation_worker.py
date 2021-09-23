import multiprocessing
import os
import logging
import queue
from ctypes import c_char_p
import random
from reprint import output
from multiprocessing import Lock, Process, Queue, current_process, Value
import time

from basic_approaches.utils.geometric_transformations import *
from basic_approaches.utils.costum_exceptions import (
    ShutdownException,
    FailedAugmentation,
)
from basic_approaches.utils.datastructures import Pixel
from basic_approaches.utils.constants import (
        Textcolor,
        ground_values,
        obstacle_values,
        person_value,
        aug_person_value)
from basic_approaches.utils.io_functions import (
    data_name,
    fg_data_loader,
    bg_data_loader,
    data_saver
)


class AugmentationWorkerManager(multiprocessing.Process):
    def __init__(
        self, num_workers, task_queue, worker_params
    ):
        multiprocessing.Process.__init__(self)
        self.exit = multiprocessing.Event()
        self.workers = []
        self.dead_workers = set()
        self.task_queue = task_queue
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            filename=os.path.join(worker_params.save_directory, "augmentation.log"),
            filemode="w",
        )
        # set up workers.
        for i in range(num_workers):
            worker = AugmentationWorker(task_queue, worker_params)
            self.workers.append(worker)
            worker.p.start()

    def run(self):
        # this process continously updates the print inside the console to show the current state for each worker
        with output(output_type="dict", interval=0) as output_lines:
            while not self.exit.is_set():
                output_lines["Images left on the queue: "] = self.task_queue.qsize()
                if len(self.dead_workers) == len(self.workers):
                    self.exit.set()
                for worker_id, worker in enumerate(self.workers, 0):
                    if worker_id in self.dead_workers:
                        continue
                    try:
                        output_lines[
                            "Worker {}".format(worker_id)
                        ] = worker.state.value.decode()
                        if worker.state.value == b"Shutdown":
                            self.dead_workers.add(worker_id)
                    except UnicodeDecodeError:
                        pass
                time.sleep(0.5)
        print("Done!")

class AugmentationWorker:
    def __init__(self, task_queue, worker_params):
        # start the image augmentation for each worker
        self.p = Process(
            target=self.augmentImage, args=(task_queue, worker_params)
        )
        # set the state as multiprocessing value so the thread can change the value of the member variable
        self.state = Value(c_char_p, b"init")

    def add_pedestrian_to_image(self, fg_img, fg_mask, bg_img, bg_mask, camera_dict, worker_params):
        # -------- Transformation/ Translation -------- #
        # Foreground fliping
        self.state.value = b"Flip..."
        flip_fg_img, flip_fg_mask = data_fliper(fg_img, fg_mask)

        # Background fliping
        flip_bg_img, flip_bg_mask = data_fliper(bg_img, bg_mask)
        self.state.value = b"Preprocess objects..."
        # Object preprocessing
        try:
            obj_img, obj_mask, obj_rect = obj_preprocesser(
                flip_fg_img, flip_fg_mask, person_value
            )
        except OSError as e:
            raise FailedAugmentation(e)
        self.state.value = b"Find random place..."
        # Random place finding
        try:
            if worker_params.force_occlusion_flag:
                bottom_pixel_person = force_occlusion(
                    flip_bg_mask,
                    obj_mask,
                    ground_values,
                    obstacle_values,
                    obj_rect.y,
                    obj_rect.h,
                    worker_params.min_occlusion_ratio,
                )
            else:
                bottom_pixel_person = random_place_finder(
                    flip_bg_mask, ground_values, obj_rect.y, obj_rect.h
                )
        except IOError:
            raise FailedAugmentation("Could not find any road to place the object on.")
        # Size of person finding
        person_height = round(
            person_height_calculation(
                camera_dict, bottom_pixel_person.x, bottom_pixel_person.y
            )
        )
        person_width = round(obj_rect.w / obj_rect.h * person_height)
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
            aug_person_value,
        )
        return fg_bg_img, fg_bg_mask, alpha_mask, bottom_pixel_person, person_width, person_height

    def augmentImageInner(self, worker_params, task_id):
        # get a random foreground and background
        fg_path = random.choice(worker_params.fg_path_list)
        bg_path = random.choice(worker_params.bg_path_list)

        # Data name chosing
        bg_name = data_name(bg_path)

        # Data loading
        self.state.value = b"loading.."
        bg_img, bg_mask, camera_dict = bg_data_loader(bg_path)
        fg_img, fg_mask = fg_data_loader(fg_path)

        fg_bg_img, fg_bg_mask, alpha_mask, bottom_pixel_person, person_width, person_height = self.add_pedestrian_to_image(fg_img, fg_mask, bg_img, bg_mask, camera_dict, worker_params)
        self.state.value = b"Saving..."
        # Data saving
        _, img_path = data_saver(
            worker_params.save_directory, bg_name, fg_bg_img, fg_bg_mask, alpha_mask, task_id
        )
        logging.debug(
            f"Saved file to {img_path} \n Params:\n Position X: {bottom_pixel_person.x} Y: {bottom_pixel_person.y}\n Object W: {person_width} H: {person_height}\n"
        )

    def augmentImage(self, task_queue, params):
        while True:
            try:
                """
                    try to get task from the queue. get_nowait() function will
                    raise queue.Empty exception if the queue is empty.
                    queue(False) function would do the same task also.
                """
                # pull a task number from the queue and try to augment the image
                task = task_queue.get_nowait()
                self.augmentImageInner(params, task)
            except queue.Empty:
                # if the queue is empty this worker can shutdown
                self.state.value = b"Shutdown"
                break
            except FailedAugmentation as e:
                # if the augmentation failed for some reason we need to put the task back on the queue
                task_queue.put_nowait(task)
                self.state.value = b"Reload..."
            except ShutdownException as e:
                # if something goes wrong the process should be closed
                self.state.value = b"Shutdown"
                break
        return True