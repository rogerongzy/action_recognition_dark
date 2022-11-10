from turtle import Vec2D
import numpy as np
import cv2
import os


def data_preparation():
    root_path = './train'
    output_root_path = './frame'
    actlist = os.listdir(root_path)
    for action in actlist:
        # print(action)

        subpath = os.path.join(root_path, action) # ./train/Drink

        videolist = os.listdir(subpath)
        for video_name in videolist:
            # print(video_name)
            output_subpath = os.path.join(output_root_path, 'v_' + action + '_g' + os.path.splitext(video_name)[0])
            os.mkdir(output_subpath)
            full_path = os.path.join(root_path, action, video_name)
            video2frame(full_path, output_subpath)



def video2frame(video_path, output_subpath):
    cap = cv2.VideoCapture(video_path)
    frame_num = 1
    while True:
        ret, frame = cap.read()
        # print(type(frame)) # numpy.ndarray
        # print(frame.shape) # (240,320,3)
        # frame_path = './video/' + str(frame_num) + '.png' # name like 1.png
        filename = 'frame' + str(frame_num).zfill(6) + '.jpg'

        output_path = os.path.join(output_subpath, filename)
        # validate_output_path = 
        
        if not ret:
            break
        
        cv2.imwrite(output_path, frame)
        frame_num += 1

    cap.release()
    cv2.destroyAllWindows()



def generate_mapping_list():
    f = open('mapping_table.txt', encoding = "utf-8")
    mapping_list = []

    line = f.readline()
    while line:
        mapping_list.append(line.split()[1])
        line = f.readline()
    f.close()

    return mapping_list



def validate_preparation(mapping_list):
    f = open('validate.txt', encoding = "utf-8")
    root_path = './validate'
    output_root_path = './validate_frame'
    os.mkdir(output_root_path)

    line = f.readline()
    while line:
        index = int(line.split()[1])
        action = mapping_list[index]
        video_name = line.split()[2]
        video_path = os.path.join(root_path, video_name)
        output_subpath = os.path.join(output_root_path, 'v_' + action + '_g' + os.path.splitext(video_name)[0])
        os.mkdir(output_subpath)
        video2frame(video_path, output_subpath)

        line = f.readline()

    f.close()



def test_preparation(mapping_list):
    f = open('test.txt', encoding = "utf-8")
    root_path = './test'
    output_root_path = './test_frame'
    os.mkdir(output_root_path)
    
    line = f.readline()
    while line:
        index = int(line.split()[1])
        action = mapping_list[index]
        video_name = line.split()[2]
        video_path = os.path.join(root_path, video_name)
        output_subpath = os.path.join(output_root_path, 'v_' + action + '_g' + os.path.splitext(video_name)[0])
        os.mkdir(output_subpath)
        video2frame(video_path, output_subpath)

        line = f.readline()

    f.close()



if __name__ == '__main__':
    mapping_list = generate_mapping_list()

    # data_preparation() # train

    # validate_preparation(mapping_list) # validate

    # test_preparation(mapping_list) # test
    
