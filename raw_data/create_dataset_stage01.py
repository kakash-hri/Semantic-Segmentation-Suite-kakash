
from glob import glob
import os
from os.path import join as fullfile
import re 
from zipfile import ZipFile
import pandas as pd
from shutil import move
from pqdm.processes import pqdm

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



def process_zip_file(seg_zip):
    # Read filenames 
    base_folder = os.path.dirname(seg_zip)
    seg_zip_name = os.path.basename(seg_zip)


    level = int(seg_zip_name[re.search("level", seg_zip_name).end()])
    drive = int(seg_zip_name[re.search("drive", seg_zip_name).end()])
    part_num = int(seg_zip_name[re.search("[0-9]*_drive", seg_zip_name).start():re.search("[0-9]*_drive", seg_zip_name).end()-6])

    id_str = "level{}_drive{}_part{}_".format(level, drive, part_num)

    if part_num != 21:
        return

    if level == 9:
        return

    video_file_path = re.sub(r'_segment_video_', '_screen_video_', seg_zip)
    video_file_path = glob(video_file_path[:re.search("_screen_video_",video_file_path).end()] + "*.avi" )[0]
    video_file = os.path.basename(video_file_path)
    
    data_csv_path = re.sub(r'_segment_video_', '_car_data_', seg_zip)
    data_csv_path = glob(data_csv_path[:re.search("_car_data_",data_csv_path).end()] + "*.csv" )[0]
    data_csv = os.path.basename(data_csv_path)


    # Find mapping between frames of seg and images
    df =pd.read_csv(fullfile(base_folder, data_csv),usecols = ['VideoFrame','SegmentFrame'])

    df.drop(df.loc[df['SegmentFrame']==-1].index, inplace=True)

    df.drop_duplicates(subset=['SegmentFrame'], keep='first', inplace=True)
    

    out_dir_images_level = fullfile(out_dir_images, "level_%d" % level)
    if not os.path.exists(out_dir_images_level):
        os.makedirs(out_dir_images_level)
    out_dir_labels_level = fullfile(out_dir_labels, "level_%d" % level)
    if not os.path.exists(out_dir_labels_level):
        os.makedirs(out_dir_labels_level)
    
    out_dir_images_part = fullfile(out_dir_images_level, "part_%d" % part_num)
    if not os.path.exists(out_dir_images_part):
        os.makedirs(out_dir_images_part)
    out_dir_labels_part = fullfile(out_dir_labels_level, "part_%d" % part_num)
    if not os.path.exists(out_dir_labels_part):
        os.makedirs(out_dir_labels_part)
    

    #Extract labels images
    print("Extracting labels")
    VideoFrame_list = []
    with ZipFile(seg_zip, 'r') as zip:
        for seg_num in df['SegmentFrame'].values:
            seg_name = seg_zip_name[:-4] + "/%d.png" % seg_num
            try:
                zip.extract(seg_name, out_dir_labels_part)
                VideoFrame_list.append(df.loc[df['SegmentFrame']==seg_num]['VideoFrame'].values[0])
            except:
                print("Error extracting %s" % seg_name)
                continue

    for seg_name in (glob(fullfile(out_dir_labels_part, seg_zip_name[:-4], "*.png"))):
        move(seg_name, out_dir_labels_part)
    
    os.rmdir(fullfile(out_dir_labels_part, seg_zip_name[:-4]))

    print("Renaming files")
    for seg_name in (glob(fullfile(out_dir_labels_part, "*.png"))):
        img_num_seg = int(seg_name[re.search("[0-9]*.png", seg_name).start():re.search("[0-9]*.png", seg_name).end() - 4])
        img_num = df.loc[df['SegmentFrame']==img_num_seg]['VideoFrame'].values[0]
        img_num_seg_str = "seg_%d" % img_num_seg
        img_num_str = "img_%d" % img_num
        os.rename(seg_name, fullfile(out_dir_labels_part, id_str + img_num_seg_str + "_" + img_num_str + "_L.png"))

    # Read video and extract images
    frame_list = ["eq(n\\,%d)" % i for i in VideoFrame_list]
    frame_list = "+".join(frame_list)
    
    print("Extracting %s to %s" % (video_file, out_dir_images_part))
    ffmpeg_cmd = "ffmpeg -i \"{video_file}\" -vf select='{frame_list}' -vsync 0 -frame_pts 1 \"{out_dir}/frames_%d.png\"".format(video_file = fullfile(base_folder, video_file), frame_list = frame_list, out_dir = out_dir_images_part)
    os.system(ffmpeg_cmd)

    # Read images based on labels
    print("Renaming files")
    for img_name in (glob(fullfile(out_dir_images_part, "*.png"))):
        img_num = int(img_name[re.search("frames_", img_name).end():re.search("frames_[0-9]*", img_name).end()])
        img_num_seg = df.loc[df['VideoFrame']==img_num]['SegmentFrame'].values[0]
        img_num_seg_str = "seg_%d" % img_num_seg
        img_num_str = "img_%d" % img_num
        os.rename(img_name, fullfile(out_dir_images_part, id_str + img_num_seg_str + "_" + img_num_str + ".png"))




if __name__ == '__main__':
    
    segmented_images_zips = search_files(".", "*_segment_video_*.zip")

    out_dir = "./created_dataset_stage01"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_dir_images = fullfile(out_dir, "images")
    if not os.path.exists(out_dir_images):
        os.makedirs(out_dir_images)
    out_dir_labels = fullfile(out_dir, "labels")
    if not os.path.exists(out_dir_labels):  
        os.makedirs(out_dir_labels)

    pqdm(segmented_images_zips, process_zip_file, n_jobs=8)
