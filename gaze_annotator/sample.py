from GazeAnnotator import GazeAnnotator
import cv2

if __name__ == '__main__':

    gaze_annotator = GazeAnnotator('class_dict.csv')
    img_rgb = cv2.imread('frame_4329_pred.png')
    eyetracking_agg_data = [[50,30],[50,30],[50,30]]
    obj, obj_prob = gaze_annotator.get_gaze_annotation(img_rgb,eyetracking_agg_data)

    print(obj)
    print(obj_prob)

