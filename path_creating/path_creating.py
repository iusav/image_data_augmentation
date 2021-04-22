
import cv2
import numpy as np
from multiprocessing import Pool
import argparse
import os, glob
import time
import sys
import json
import csv
import matplotlib.pyplot as plt 
from tqdm import tqdm

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

def pathWriter(data, path):
    # Write paths to a CSV file
    with open(path, "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in data:
            writer.writerow(line)
def pathCreater(json_path):
    data_name = '_'.join(json_path.split('/')[-1].split('_')[:-2])
    mask_directory = '/'.join(json_path.split('/')[:-1])
    img_directory =  mask_directory.replace('gtFine/train', 'leftImg8bit/train')
    mask_path = os.path.join(mask_directory, str(data_name) + '_gtFine_labelIds.png')
    img_path = os.path.join(img_directory, str(data_name) + '_leftImg8bit.png')
    
    return img_path, mask_path

def processJsonFiles(json_path, obj_bg_ratio=0.001):
    fg_path = None
    bg_path = None
    with open(json_path) as json_file:
        json_data = json.load(json_file)
        obj_key = 'objects'
        if obj_key in  json_data:
            if ('imgHeight' in json_data) & ('imgWidth' in json_data):
                imgArea = int(json_data['imgHeight'] * json_data['imgWidth'])
            else:
                imgArea = 2097152

            for label in json_data[obj_key]:
                if any([className in label['label'] for className in fgNames]):
                    obj_polygon = np.expand_dims(np.array(label['polygon']), axis = 1)
                    objArea = cv2.contourArea(obj_polygon)
                    current_obj_bg_ratio = objArea/imgArea
                    if current_obj_bg_ratio >= obj_bg_ratio:
                        img_path, mask_path = pathCreater(json_path)
                        act_data = [img_path, mask_path]
                        fg_path = act_data

                if any([className in label['label'] for className in bgNames]):
                    img_path, mask_path = pathCreater(json_path)
                    act_data = [img_path, mask_path]
                    bg_path = act_data
    return fg_path, bg_path

def json_files_multiprocessing(func, argument_list, num_processes):

    pool = Pool(processes=num_processes)

    fgList = []
    bgList = []
    for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list), desc="Scanning scene for pedestrians and roads: "):
        if result[0]:
            fgList.append(result[0])
        if result[1]:
            bgList.append(result[1])
    return fgList, bgList

def glob_multiprocessing(func, argument_list, num_processes):

    pool = Pool(processes=num_processes)

    result_list_tqdm = []
    for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list), desc="Globbing all json files: "):
        result_list_tqdm.extend(result)

    return result_list_tqdm

def globJsonFiles(jsonDir):
    return glob.glob(os.path.join(jsonDir, "*.json"))
# MAIN
if __name__ == '__main__':
    start_time = time.time()
    fgNames = ['person']
    bgNames = ['ground','road','sidewalk']
    fgFileName = "citysc_fgPaths.csv"
    bgFileName = "citysc_bgPaths.csv"
    file_dir = os.path.dirname(os.path.realpath(__file__))
    DEFAULT_OUTPUT = os.path.abspath(os.path.join(file_dir, "../basic_approaches/"))
    DEFAULT_JSON_DIR = '/mrtstorage/datasets/public/cityscapes/gtFine'
    DEFAULT_NUMBER_PROCESSES = 16

    parser = argparse.ArgumentParser(description='Create the paths needed for the basic augmentation script.')
    parser.add_argument('--output-dir', dest='output_dir', default=DEFAULT_OUTPUT, help='Choose the destination where you want to save the output csv to')
    parser.add_argument('--json-dir', dest='json_dir', default=DEFAULT_JSON_DIR, help='Provide the PATH_TO_CITYSCAPES/cityscapes/gtFine')
    parser.add_argument('--processes', dest='processes', type=int, default=DEFAULT_NUMBER_PROCESSES, help='Set the number of processes for multiprocessing')

    args = parser.parse_args()

    output_dir = args.output_dir
    fgPaths = os.path.join(output_dir, fgFileName)
    bgPaths = os.path.join(output_dir, bgFileName)
    json_directory = args.json_dir
    num_processes = args.processes

    glob_dirs = glob.glob(os.path.join(json_directory, '*','*'))
    json_paths = glob_multiprocessing(globJsonFiles, glob_dirs, num_processes)
    if not json_paths:
        print(f"{textcolor.WARNING}Warning{textcolor.ENDC}: There were no csv files were I assumed them to be. Is {os.path.abspath(json_directory)} really the right path to cityscapes?")
        sys.exit(0)
    # Ratio between object and background area
    # Min ratio for object chosing
    # obj_bg_ratio = 0 means chosing objects with different size
    # can change obj_bg_ratio, for Example obj_bg_ratio = 0.01
    obj_bg_ratio = 0.001    #obj/img

    # I don't know how to give more than one argument at the moment
    # so obj_bg_ratio has a default value in the function definition
    fgList, bgList  = json_files_multiprocessing(processJsonFiles, json_paths, num_processes)
    if not fgList and not bgList:
        print(f"{textcolor.WARNING}Warning{textcolor.ENDC}: I found some csv, but it seems like I could not retrieve any path to the cityscapes images. Maybe {os.path.abspath(json_directory)} was not the right path to cityscapes.") 
        sys.exit(1)
    pathWriter(fgList, fgPaths)
    pathWriter(bgList, bgPaths)
    
    print('FG data: ',len(fgList))
    print('BG data: ',len(bgList))
    
    print("------------------- %s done --------------------" % (max(len(fgList), len(bgList))))
    print("----------------- %s seconds ----------------" % ( round((time.time() - start_time), 2) ))