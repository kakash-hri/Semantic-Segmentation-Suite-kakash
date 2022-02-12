#%%
from glob import glob
import os
from os.path import join as fullfile
import re 
from zipfile import ZipFile
import pandas as pd
from shutil import copyfile
import cv2
import json
import numpy as np

in_dir = "./created_dataset_stage02_1280x720"
out_dir = ".."

in_datasets = glob(fullfile(in_dir, '*/'))


# %%

for in_dataset in in_datasets:

    dataset_name = 'dataset_1280x720_' + os.path.basename(in_dataset[:-1])
    all_files = glob(fullfile(in_dataset, 'images/*'))

    perms = np.random.permutation(len(all_files))

    train_idxs = perms[:int(len(perms) * 0.7)]
    val_idxs = perms[int(len(perms) * 0.7):int(len(perms) * 0.9)]
    test_idxs = perms[int(len(perms) * 0.9):]

    train_path = fullfile(out_dir, dataset_name, 'train')
    train_labels_path = fullfile(out_dir, dataset_name, 'train_labels')
    val_path = fullfile(out_dir, dataset_name, 'val')
    val_labels_path = fullfile(out_dir, dataset_name, 'val_labels')
    test_path = fullfile(out_dir, dataset_name, 'test')
    test_labels_path = fullfile(out_dir, dataset_name, 'test_labels')

    for pths in [train_path, train_labels_path, val_path, val_labels_path, test_path, test_labels_path]:
        if not os.path.exists(pths):
            os.makedirs(pths)

    for idx in train_idxs:
        filename = all_files[idx]
        copyfile(filename, fullfile(train_path, os.path.basename(filename)))
        filename_labels = filename.replace('images', 'labels').replace('.png', '_L.png')
        copyfile(filename_labels, fullfile(train_labels_path, os.path.basename(filename_labels)))

    for idx in val_idxs:
        filename = all_files[idx]
        copyfile(filename, fullfile(val_path, os.path.basename(filename)))
        filename_labels = filename.replace('images', 'labels').replace('.png', '_L.png')
        copyfile(filename_labels, fullfile(val_labels_path, os.path.basename(filename_labels)))

    for idx in test_idxs:
        filename = all_files[idx]
        copyfile(filename, fullfile(test_path, os.path.basename(filename)))
        filename_labels = filename.replace('images', 'labels').replace('.png', '_L.png')
        copyfile(filename_labels, fullfile(test_labels_path, os.path.basename(filename_labels)))
        
    copyfile(fullfile(in_dataset, 'class_dict.csv'), fullfile(out_dir, dataset_name, 'class_dict.csv'))
