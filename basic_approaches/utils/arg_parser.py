import os
import argparse
def create_parser(file_dir):
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
	return parser
