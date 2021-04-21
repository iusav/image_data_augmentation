
import cv2
import numpy as np
import os, glob
import time
import json
import csv
import matplotlib.pyplot as plt 
from tqdm import tqdm

fgNames = ['person']
bgNames = ['ground','road','sidewalk']
json_directory = '/mrtstorage/datasets/public/cityscapes/gtFine'
fgPaths = '/home/roesch/Data/Documents/lehre/students/Anton/image_data_augmentation/basic_approaches/citysc_fgPaths.csv'
bgPaths = '/home/roesch/Data/Documents/lehre/students/Anton/image_data_augmentation/basic_approaches/citysc_bgPaths.csv'
json_paths = glob.glob(os.path.join(json_directory, '*','*','*.json'))
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
# MAIN
if __name__ == '__main__':
    start_time = time.time()
    
    dataNumb   = 0
    fgCounter  = 0
    bgCounter  = 0
    fgList     = []
    bgList     = []
    # Ratio between object and background area
    # Min ratio for object chosing
    # obj_bg_ratio = 0 means chosing objects with different size
    # can change obj_bg_ratio, for Example obj_bg_ratio = 0.01
    obj_bg_ratio = 0.001    #obj/img
    for json_path in tqdm(json_paths):
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
                            fgList.append(act_data)
                            fgCounter += 1
                                           
                    if any([className in label['label'] for className in bgNames]):              
                        img_path, mask_path = pathCreater(json_path)
                        act_data = [img_path, mask_path]
                        bgList.append(act_data)
                        bgCounter += 1

        dataNumb += 1
    
    pathWriter(fgList, fgPaths)
    pathWriter(bgList, bgPaths)
    
    print('FG data: ',fgCounter)
    print('BG data: ',bgCounter)
    
    print("------------------- %s done --------------------" % (dataNumb))
    print("----------------- %s seconds ----------------" % ( round((time.time() - start_time), 2) ))