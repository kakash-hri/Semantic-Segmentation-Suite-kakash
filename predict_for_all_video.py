
#%%
from glob import glob
import os
from os.path import join as fullfile
import re 
from zipfile import ZipFile
from shutil import move, rmtree, make_archive
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



# for i, video_file in enumerate(video_files):
def process_video(args_array):
    i = args_array[0]
    video_file = args_array[1]
    base_folder = os.path.dirname(video_file)
    video_file_name = os.path.basename(video_file)


    level = int(video_file_name[re.search("level", video_file_name).end()])
    drive = int(video_file_name[re.search("drive", video_file_name).end()])
    part_num = int(video_file_name[re.search("[0-9]*_drive", video_file_name).start():re.search("[0-9]*_drive", video_file_name).end()-6])

    # %%
    # if level == 9:
    #     return

    out_dir = './segmentation_results'
    if os.path.exists(out_dir) == False:
        os.mkdir(out_dir)

    out_dir_part = fullfile(out_dir, 'part_{}'.format(part_num))
    if os.path.exists(out_dir_part) == False:
        os.mkdir(out_dir_part)

    out_dir_seg = fullfile(out_dir_part, video_file_name[:-4])
    if os.path.exists(out_dir_seg) == False:
        os.mkdir(out_dir_seg)

    ffmpeg_cmd = 'ffmpeg -i "{}" "{}/frame_%04d.png"'.format(video_file, out_dir_seg)
    os.system(ffmpeg_cmd)

    if level == 1 or level == 5:
        level_str = 'level_1_5'
    elif level == 2 or level == 8:
        level_str = 'level_2_8'
    elif level == 3 or level == 6:
        level_str = 'level_3_6'
    elif level == 4 or level == 7:
        level_str = 'level_4_7'

    model_name = "DeepLabV3_plus"
    ckpt_name = "DeepLabV3_plus_dataset_1280x720_" + level_str 
    dataset_name = "dataset_1280x720_" + level_str

    width = 1280
    height = 720
    gpu = i%2

    pred_cmd = 'python ./predict_folder.py --image "{}" --checkpoint_path "./{}_checkpoints/latest_model_{}.ckpt" --model {} --dataset "{}" --resize {}x{} --gpu {}  --crop_height {} --crop_width {}'.format(out_dir_seg, ckpt_name, ckpt_name, model_name, dataset_name, width, height, gpu, height, width)
    os.system(pred_cmd)

    rmtree(out_dir_seg)

    curr_dir = os.getcwd()
    os.chdir(out_dir_part)

    zip_cmd = 'zip -r "{}" "{}"'.format(video_file_name[:-4] + '.zip', video_file_name[:-4] + "_preds")
    os.system(zip_cmd)
    rmtree(video_file_name[:-4] + "_preds")

    os.chdir(curr_dir)


if __name__ == '__main__':
    video_files = search_files('./raw_data/', '*_screen_video_*.avi')
    video_files = [video_file for video_file in video_files if 'level9' not in video_file]

    list(map(process_video, enumerate(video_files)))
    # pqdm(enumerate(video_files), process_video,n_jobs=4)