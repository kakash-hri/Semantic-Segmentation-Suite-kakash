
#%%
from glob import glob
import os
from os.path import join as fullfile
import re 
from zipfile import ZipFile
import pandas as pd
from shutil import move
from pqdm.processes import pqdm
import cv2
import json
import numpy as np

def search_files(search_path, search_file):
    # print("Searching {search_path} for {search_file}".format(search_path = search_path, search_file = search_file))
    found_files = []    
    files_got = glob(fullfile(search_path, search_file))
    if files_got:
        found_files.extend(files_got)

    search_folders = glob(fullfile(search_path, '*/'))
    for folder in search_folders:
        found_files.extend(search_files(folder,search_file))
    return found_files



def process_image(args):
    image_name, people_bgrs, car_bgrs, out_dir_lvl = args
    img = cv2.imread(image_name)
    img_resized = cv2.resize(img, (512,512), interpolation=cv2.INTER_NEAREST)

    for people_bgr in people_bgrs:
        img_resized[np.where((img_resized==people_bgr).all(axis=2))] = people_bgrs[0]
    for car_bgr in car_bgrs:
        img_resized[np.where((img_resized==car_bgr).all(axis=2))] = car_bgrs[0]
    cv2.imwrite(fullfile(out_dir_lvl, os.path.basename(image_name)), img_resized)




# %%
if __name__ == '__main__':

    level_grps = [[1,5],
                [2,8],
                [3,6],
                [4,7]] 

    for level_grp in level_grps:

        json_file = "In-person Study/objID_2_rgb_map.json"
        with open(json_file) as f:
            obj_id_to_rgb = json.load(f)

        objID2obj = {0: 'Sky',
                    254: 'Road',
                    253: 'Sidewalk',
                    252: 'TrafficSignal',
                    251: 'RoadSign',
                    250: 'ParkedCars',
                    249: 'StaticPeople',
                    248: 'InsideCar',
                    247: 'Speedometer',
                    246: 'Trees',
                    245: 'Benches',
                    244: 'Bushes',
                    243: 'Buildings',
                    242: 'Trashcans',
                    241: 'Fences',
                    240: 'StreetLamp',
                    239: 'BusStop',
                    238: 'Background'}
        for i in range(100):
            objID2obj[1+i] = 'People_%d'%(i+1)
            objID2obj[101+i] = 'Car_%d'%(i+1)

        obj2bgr = {}
        people_bgrs = []
        car_bgrs = []

        for objID,obj in objID2obj.items():
            obj2bgr[obj] = eval(obj_id_to_rgb[str(objID)])

            if obj.startswith('People') or obj.startswith('StaticPeople'):
                people_bgrs.append(obj2bgr[obj])
            elif obj.startswith('Car') or obj.startswith('ParkedCars'):
                car_bgrs.append(obj2bgr[obj])



        out_dir_base = "./created_dataset_stage02"
        if not os.path.exists(out_dir_base):
            os.makedirs(out_dir_base)

        out_dir = fullfile(out_dir_base, "level_%d_%d" % (level_grp[0], level_grp[1]))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)


        obj2rgb = {}
        obj2rgb["People"] = people_bgrs[0]
        obj2rgb["Car"] = car_bgrs[0]
        for obj,bgr in obj2bgr.items():
            if not (obj.startswith('People') or obj.startswith('Car')) or obj.startswith('StaticPeople') or obj.startswith('ParkedCars'):
                obj2rgb[obj] = (bgr[2],bgr[1],bgr[0])

        #Create csv file
        csv_file = fullfile(out_dir, "class_dict.csv")
        with open(csv_file, 'w') as f:
            f.write("name,r,g,b\n")
            for obj, rgb in obj2rgb.items():
                f.write("%s,%d,%d,%d\n"%(obj, rgb[0], rgb[1], rgb[2]))

        in_dir = "./created_dataset_stage01"
        for lvl in level_grp:
            for lab_or_img in ['labels', 'images']:
                in_dir_lvl = fullfile(in_dir,lab_or_img, "level_%d" % lvl)
                in_dir_lvl = glob(fullfile(in_dir_lvl, "part_*"))[0]

                out_dir_lvl = fullfile(out_dir, lab_or_img)
                if not os.path.exists(out_dir_lvl):
                    os.makedirs(out_dir_lvl)

                image_names = glob(fullfile(in_dir_lvl, "*.png"))
                # for image_name in image_names: #resize images using cv2.resize

                image_names_args = [[image_name, people_bgrs, car_bgrs, out_dir_lvl] for image_name in image_names]
                print("Processing %d images from %s "%(len(image_names_args), in_dir_lvl))

                pqdm(image_names_args, process_image, n_jobs=24)


