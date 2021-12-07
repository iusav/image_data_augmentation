
import os
import sys
from multiprocessing import Queue
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from basic_approaches.utils.io_functions import (
    path_reader,
    current_id,
    check_for_folder,
)
from basic_approaches.utils.augmentation_worker import AugmentationWorkerManager
from basic_approaches.utils.datastructures import AugmentationWorkerParams
from basic_approaches.utils.arg_parser import create_parser


if __name__ == "__main__":
    file_dir = os.path.dirname(os.path.realpath(__file__))
    parser = create_parser(file_dir)
    args = parser.parse_args()
    
    check_for_folder(args.output_path)
    fg_path_list = path_reader(args.fg_paths)
    bg_path_list = path_reader(args.bg_paths)
    
    if (args.annotat_status != 'mask') & (args.annotat_status != 'polygon'):
        print('! Warning !')
        print('Entered "annotation status" into the arguments: ',args.annotat_status)
        print('"Annotation status" is not "mask" or "polygon"')
        print('Annotation status was automatically changed to "mask"')
        args.annotat_status = 'mask'
    else:
        print('Entered "annotation status" into the arguments: ',args.annotat_status)

    id_data = current_id(args.output_path)
    # id_data = 1
    if id_data >= args.dataset_size:
        print(
            "There are already enough images in the dataset. Either increase the dataset size or delete some images."
        )
        sys.exit(0)
    print(f"Start at image id {id_data}")
    task_queue = Queue()
    # put as many task on the queue as we want to have images in our dataset
    for i in range(args.dataset_size - id_data):
        task_queue.put(i)
    worker_params = AugmentationWorkerParams(args.output_path, fg_path_list, bg_path_list, args.force_occlusion, args.min_occlusion_ratio, args.annotat_status)
    manager = AugmentationWorkerManager(
        args.num_processes, task_queue, worker_params
    )
    manager.start()
    sys.exit(0)