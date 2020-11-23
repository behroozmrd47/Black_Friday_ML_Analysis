import os
import cv2
import h5py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import src.Constants as Cns
from src.utils.utility_functions import stack_images, image_set_preparation


# def image_set_preparation(image_set, resize_target=(512, 640), crop_width_rate=0.8):
#     image_set_modified = []
#     for image in image_set:
#         image_resized = cv2.resize(image, dsize=resize_target, interpolation=cv2.INTER_CUBIC)
#         crop_rate_fr_edge = (1.0 - crop_width_rate) / 2
#         img_wid = image_resized.shape[0]
#         image_resized_cropped = image_resized[
#                                 int(img_wid * crop_rate_fr_edge) + 1:int(img_wid * (1 - crop_rate_fr_edge))]
#         image_set_modified.append(image_resized_cropped)
#     return np.array(image_set_modified)


def process_concat_images():
    included_groups = ['Female_Negative', 'Male_Negative', 'Female_Positive', 'Male_Positive']
    # included_groups = ['Male_Negative', 'Female_Positive', 'Male_Positive']
    for dir in os.listdir(Cns.RAW_DATA_FOLDER):
        if dir in included_groups:
            print(dir)
            load_adr = os.path.join(Cns.RAW_DATA_FOLDER, dir)
            all_image_array = []
            all_label_array = []
            for file_name in os.listdir(load_adr):
                file_load_addr = os.path.join(load_adr, file_name)
                hdf5_file = h5py.File(file_load_addr, "r")

                image_array = np.array(hdf5_file['Image_Ori'])
                image_array_modified = image_set_preparation(image_array)
                all_image_array.extend(image_array_modified)

                label_array = np.array(hdf5_file['Mask'])
                label_array_modified = image_set_preparation(label_array)
                all_label_array.extend(label_array_modified)
            processed_dic = {'feature': all_image_array, 'label': all_label_array}
            write_addr = os.path.join(Cns.PROCESSED_DATA_FOLDER, dir + '.pcl')
            with open(write_addr, 'wb') as file_handler:
                pickle.dump(processed_dic, file_handler)


def process_concat_images_Cardiac(input_data_folder_adr, included_groups=Cns.INCLUDED_GROUPS, dtype=np.int16):
    image_array_all = np.empty((0, Cns.IMG_DEPTH, Cns.IMG_ROWS, Cns.IMG_COLS), dtype=dtype)
    label_array_all = []

    for dir in os.listdir(input_data_folder_adr):
        if dir in included_groups:
            print(dir)
            load_adr = os.path.join(input_data_folder_adr, dir)
            count = 0
            for file_name in os.listdir(load_adr):
                file_load_addr = os.path.join(load_adr, file_name)
                hdf5_file = h5py.File(file_load_addr, "r")
                print(file_name, hdf5_file['Image_Ori'].shape)

                image_array = np.array(hdf5_file['Image_Ori'])
                image_array_modified = image_set_preparation(image_array)
                image_array_stacked = stack_images(image_array_modified, Cns.IMG_DEPTH)
                image_array_all = np.concatenate((image_array_all, image_array_stacked), axis=0)

                if (dir == 'Cardiac_Mat'):
                    for i in range(image_array_stacked.shape[0]):
                        label_array_all.append(1)
                else:
                    for i in range(image_array_stacked.shape[0]):
                        label_array_all.append(0)
                count = count + 1
                if (dir == 'Cardiac_Mat'):
                    continue
                elif (count > 5):
                    break

    image_array_all = np.expand_dims(image_array_all, axis=4).astype(dtype=dtype)
    label_array_all = np.array(label_array_all)
    return image_array_all, label_array_all


def process_concat_images_3d(input_data_folder_adr, included_groups=Cns.INCLUDED_GROUPS, dtype=np.int16):
    """
    To load and process mat files. Mat files should be save with version 7.3 out of Matlab.
    :param input_data_folder_adr: input data folder
    :param included_groups: groups to be processed
    :param dtype: output datatype
    :return: image and lacal inputed, stacked
    """
    image_array_all = np.empty((0, Cns.IMG_DEPTH, Cns.IMG_ROWS, Cns.IMG_COLS), dtype=dtype)
    label_array_all = np.empty((0, Cns.IMG_DEPTH, Cns.IMG_ROWS, Cns.IMG_COLS), dtype=dtype)

    for dir in os.listdir(input_data_folder_adr):
        if dir in included_groups:
            print(dir)
            load_adr = os.path.join(input_data_folder_adr, dir)
            for file_name in os.listdir(load_adr):
                file_load_addr = os.path.join(load_adr, file_name)
                hdf5_file = h5py.File(file_load_addr, "r")
                print(file_name, hdf5_file['Image_Ori'].shape)

                image_array = np.array(hdf5_file['Image_Ori'])
                image_array_modified = image_set_preparation(image_array)
                image_array_stacked = stack_images(image_array_modified, Cns.IMG_DEPTH)
                image_array_all = np.concatenate((image_array_all, image_array_stacked), axis=0)

                label_array = np.array(hdf5_file['Mask'])
                label_array_modified = image_set_preparation(label_array)
                label_array_stacked = stack_images(label_array_modified, Cns.IMG_DEPTH)
                label_array_all = np.concatenate((label_array_all, label_array_stacked), axis=0)
                if not(image_array.shape == label_array.shape):
                    print('********************* Warning *********************')
                    print(image_array.shape, label_array.shape)

    image_array_all = np.expand_dims(image_array_all, axis=4).astype(dtype=dtype)
    label_array_all = np.expand_dims(label_array_all, axis=4).astype(dtype=dtype)
    return image_array_all, label_array_all


def process_concat_images_2d(input_data_folder_adr, included_groups=Cns.INCLUDED_GROUPS, dtype=np.int16):
    image_array_all = np.empty((0, Cns.IMG_ROWS, Cns.IMG_COLS))
    label_array_all = np.empty((0, Cns.IMG_ROWS, Cns.IMG_COLS))
    for dir in os.listdir(input_data_folder_adr):
        if dir in included_groups:
            print(dir)
            load_adr = os.path.join(input_data_folder_adr, dir)
            for file_name in os.listdir(load_adr):
                file_load_addr = os.path.join(load_adr, file_name)
                hdf5_file = h5py.File(file_load_addr, "r")
                print(file_name, hdf5_file['Image_Ori'].shape)

                image_array = np.array(hdf5_file['Image_Ori'])
                image_array_modified = image_set_preparation(image_array)
                image_array_all = np.concatenate((image_array_all, image_array_modified), axis=0)

                label_array = np.array(hdf5_file['Mask'])
                label_array_modified = image_set_preparation(label_array)
                label_array_all = np.concatenate((label_array_all, label_array_modified), axis=0)

    image_array_all = np.expand_dims(image_array_all, axis=-1).astype(dtype=dtype)
    image_array_all /= 255.0
    label_array_all = np.expand_dims(label_array_all, axis=-1).astype(dtype=dtype)
    return image_array_all, label_array_all


# def process_concat_images(input_data_folder_adr, dtype=np.int16, included_groups=Cns.INCLUDED_GROUPS,
#                           test_train_ratio=Cns.TEST_RATIO):
#     train_image_array = np.empty((0, Cns.IMG_DEPTH, Cns.IMG_ROWS, Cns.IMG_COLS, 1), dtype=dtype)
#     train_label_array = np.empty((0, Cns.IMG_DEPTH, Cns.IMG_ROWS, Cns.IMG_COLS, 1), dtype=dtype)
#     test_image_array = np.empty((0, Cns.IMG_DEPTH, Cns.IMG_ROWS, Cns.IMG_COLS, 1), dtype=dtype)
#     test_label_array = np.empty((0, Cns.IMG_DEPTH, Cns.IMG_ROWS, Cns.IMG_COLS, 1), dtype=dtype)
#     for dir in os.listdir(input_data_folder_adr):
#         if dir in included_groups:
#             print(dir)
#             load_adr = os.path.join(input_data_folder_adr, dir)
#             dir_image_array = np.empty((0, Cns.IMG_DEPTH, Cns.IMG_ROWS, Cns.IMG_COLS))
#             dir_label_array = np.empty((0, Cns.IMG_DEPTH, Cns.IMG_ROWS, Cns.IMG_COLS))
#             for file_name in os.listdir(load_adr):
#                 file_load_addr = os.path.join(load_adr, file_name)
#                 hdf5_file = h5py.File(file_load_addr, "r")
#                 print(file_name, hdf5_file['Image_Ori'].shape)
#
#                 image_array = np.array(hdf5_file['Image_Ori'])
#                 image_array_modified = image_set_preparation(image_array)
#                 image_array_stacked = stack_images(image_array_modified, Cns.IMG_DEPTH)
#                 dir_image_array = np.concatenate((dir_image_array, image_array_stacked), axis=0)
#
#                 label_array = np.array(hdf5_file['Mask'])
#                 label_array_modified = image_set_preparation(label_array)
#                 label_array_stacked = stack_images(label_array_modified, Cns.IMG_DEPTH)
#                 dir_label_array = np.concatenate((dir_label_array, label_array_stacked), axis=0)
#
#             dir_image_array = np.expand_dims(dir_image_array, axis=4).astype(dtype=dtype)
#             dir_label_array = np.expand_dims(dir_label_array, axis=4).astype(dtype=dtype)
#
#             x_train, x_test, y_train, y_test = train_test_split(dir_image_array, dir_label_array,
#                                                                 test_size=test_train_ratio, shuffle=False)
#             train_image_array = np.concatenate((train_image_array, x_train), axis=0)
#             test_image_array = np.concatenate((test_image_array, x_test), axis=0)
#             train_label_array = np.concatenate((train_label_array, y_train), axis=0)
#             test_label_array = np.concatenate((test_label_array, y_test), axis=0)
#     return train_image_array, train_label_array, test_image_array, test_label_array


if __name__ == '__main__':
    process_concat_images()
