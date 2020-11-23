import numpy as np
import os
from pathlib import Path
import shutil
import src.Constants as Cns
import cv2


def image_set_preparation(image_set, resize_target=(Cns.IMG_COLS, Cns.IMG_ROWS), crop_width_rate=0.8):
    cv_width_target = resize_target[0]
    cv_height_target = int(resize_target[1] / crop_width_rate)

    image_set_modified = []
    for image in image_set:
        image_resized = cv2.resize(image, dsize=(cv_width_target, cv_height_target), interpolation=cv2.INTER_CUBIC)
        crop_rate_fr_edge = (1.0 - crop_width_rate) / 2
        img_wid = image_resized.shape[0]
        image_resized_cropped = image_resized[
                                int(img_wid * crop_rate_fr_edge) + 1:int(img_wid * (1 - crop_rate_fr_edge))]
        image_set_modified.append(image_resized_cropped)
    return np.array(image_set_modified)

def binarize_images(images_array, threshold=0.2):
    return (images_array > threshold) * 1.0


def stack_images(image_array, stack_count):
    mbs = stack_count
    mbsh = stack_count // 2
    x = image_array.shape[1]
    y = image_array.shape[2]
    length = int(len(image_array) / mbsh) - 1
    images_stacked = np.empty((length, mbs, x, y), dtype=image_array.dtype)
    for i in range(length):
        images_stacked[i, :, :, :] = image_array[i * mbsh:i * mbsh + mbs, :, :]
    return images_stacked


def check_exist_folder(folder_path, create_if_not_exist=True):
    is_exist = os.path.exists(folder_path)
    if not is_exist and create_if_not_exist:
        os.makedirs(folder_path)
        return create_if_not_exist
    return is_exist


def de_stack_images(stacked_image_array, stack_num):
    stacked_image_array = np.squeeze(stacked_image_array, -1)
    image_array = stacked_image_array[0]
    for i in range(1, stacked_image_array.shape[0]):
        image_array = np.concatenate((image_array, stacked_image_array[i, stack_num // 2: stack_num]))
    return image_array


def rmdir(directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


def delete_folder(folder_address):
    folder_address_abs = os.path.abspath(folder_address)
    shutil.rmtree(folder_address_abs, ignore_errors=True)
    # logger.info('local folder %s is deleted.' % folder_address_abs)


if __name__ == '__main__':
    bucket = 'ctra-test'
    key = 'DICOM-INPUT/user/upload-id'
    downloadPath = 'tmp/MODULE2-OUTPUT/Behrooz/test'
    Path(downloadPath).mkdir(parents=True)
    download_s3_folder(bucket, key, downloadPath)
