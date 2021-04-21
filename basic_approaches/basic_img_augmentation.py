import os
import sys
import cv2
import csv
import argparse
import glob
import time
import random
import numpy as np
import matplotlib.pyplot as plt 
from geometric_transformations import *

# Background
road_value = 7
ground_value = 6
sidewalk_value = 8

# Foreground
person_value = 24


def pathReader(path):
    # Read paths of a CSV file
    with open(path, newline='') as fg_bg_data:
        reader = csv.reader(fg_bg_data)
        data = list(reader)
    return data

def data_name(fg_path,bg_path): 
    FGname = '_'.join(fg_path[1].split('/')[-1].split('_')[:-2])
    BGname = '_'.join(bg_path[1].split('/')[-1].split('_')[:-2])
    
    return FGname, BGname

def data_loader(fg_path,bg_path):    
    # Foreground paths
    imgFG_path = fg_path[0]; maskFG_path = fg_path[1]
    
    # Background paths
    imgBG_path = bg_path[0]; maskBG_path = bg_path[1]
    
    
    FGimg = cv2.imread(imgFG_path)
    FGimg = cv2.cvtColor(FGimg, cv2.COLOR_BGR2RGB)
    FGmask = cv2.imread(maskFG_path)
    FGmask = cv2.cvtColor(FGmask, cv2.COLOR_BGR2RGB)
    
    BGimg = cv2.imread(imgBG_path); BGimg = cv2.cvtColor(BGimg, cv2.COLOR_BGR2RGB)
    BGmask = cv2.imread(maskBG_path); BGmask = cv2.cvtColor(BGmask, cv2.COLOR_BGR2RGB)
    
    return FGimg, FGmask, BGimg, BGmask   

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

if __name__ == '__main__':
    file_dir = os.path.dirname(os.path.realpath(__file__))
    DEFAULT_DATASET_SIZE = 10
    DEFAULT_OUTPUT_PATH = os.path.abspath(os.path.join(file_dir, "../created_dataset"))
    DEFAULT_FG_PATH = os.path.abspath(os.path.join(file_dir, "../basic_approaches/citysc_fgPaths.csv"))
    DEFAULT_BG_PATH = os.path.abspath(os.path.join(file_dir, "../basic_approaches/citysc_bgPaths.csv"))

    parser = argparse.ArgumentParser(description='Create the augmented dataset for cityscapes.')
    parser.add_argument('--dataset-size', dest='dataset_size', type=int, default=DEFAULT_DATASET_SIZE,
                        help='Choose the size of the created dataset')
    parser.add_argument('--output-path', dest='output_path', default=DEFAULT_OUTPUT_PATH, help='Choose where the created images are saved to.')
    parser.add_argument('--fg', dest='fgPaths', default=DEFAULT_FG_PATH, help='Select the csv files which where created by the path_creating script')
    parser.add_argument('--bg', dest='bgPaths', default=DEFAULT_BG_PATH, help='Select the csv files which where created by the path_creating script')
    args = parser.parse_args()
    dataset_size = args.dataset_size

    fgPaths = args.fgPaths
    bgPaths = args.bgPaths
    save_directory = args.output_path

    fg_path_list = pathReader(fgPaths)
    bg_path_list = pathReader(bgPaths)

    start_time = time.time()

    id_data = current_id()
    #id_data = 1
    
    while  (id_data <= dataset_size):
            fg_path = random.choice(fg_path_list)  
            bg_path = random.choice(bg_path_list)

            # Data name chosing
            FGname, BGname = data_name(fg_path, bg_path)

            # Data loading
            print("loading..")
            try:
                FGimg, FGmask, BGimg, BGmask = data_loader(fg_path, bg_path)
            except:
                continue
            print("Foreground:" + fg_path[0])
            print("Background:" + bg_path[0])
            FGheight = FGmask.shape[0]; FGwidth = FGmask.shape[1]
            BGheight = BGmask.shape[0]; BGwidth = BGmask.shape[1]


            # -------- Transformation/ Translation -------- #
            # Foreground fliping
            print("Flip...")
            flip_FGimg, flip_FGmask = data_fliper(FGimg, FGmask)

            # Background fliping
            flip_BGimg, flip_BGmask = data_fliper(BGimg, BGmask)
            print("Preprocess objects..")
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
                continue
            print("Find random place...")
            # Random place finding
            stand_y, stand_x = random_place_finder(flip_BGmask,
                                                   ground_value, 
                                                   sidewalk_value, 
                                                   road_value, 
                                                   BGheight, 
                                                   BGwidth,
                                                   y,
                                                   h)
        
            # Size of person finding
            obj_mask_height = obj_mask.shape[0]; obj_mask_width = obj_mask.shape[1]
            stand_obj_height, stand_obj_width = person_size_finder(stand_y, 
                                                                   w, 
                                                                   h, 
                                                                   obj_mask_height, 
                                                                   obj_mask_width)
            print("Matting..")
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
                                                       stand_y, stand_x, 
                                                       stand_obj_height, 
                                                       stand_obj_width, 
                                                       BGheight, 
                                                       BGwidth,
                                                       person_value)
            print("Saving...")
            # Data saving 
            data_saver(BGname, fg_bg_img, fg_bg_mask, id_data)

            
            #plt.imshow(fg_bg_img)  
            #plt.show()

            #plt.imshow(fg_bg_mask)
            #plt.show() 


            print("------------------- %s done --------------------" % (id_data))
            print("------------------- %s rest --------------------" % (dataset_size - id_data))
            id_data += 1
    print("----------------- %s seconds ----------------" % ( round((time.time() - start_time), 2) ))