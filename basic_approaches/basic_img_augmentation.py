
import os
import sys
import argparse
from multiprocessing import Queue
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from basic_approaches.utils.io_functions import (
    path_reader,
    current_id,
    check_for_folder,
)
from basic_approaches.utils.augmentation_worker import AugmentationWorkerManager
from basic_approaches.utils.datastructures import AugmentationWorkerParams


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
    worker_params = AugmentationWorkerParams(save_directory, fg_path_list, bg_path_list, force_occlusion_flag, min_occlusion_ratio)
    manager = AugmentationWorkerManager(
        num_processes, task_queue, worker_params
    )
    manager.start()
    while not task_queue.empty and len(manager.workers) > 1:
        print("Done!")
    sys.exit(0)
