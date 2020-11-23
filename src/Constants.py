import numpy as np

RUN_NAME_2D = 'FF_All_Nov19'
RUN_NAME_3D = 'FF_All_Nov18'


INCLUDED_GROUPS = ['Male_Positive_Limited']
# INCLUDED_GROUPS = ['Female_Negative', 'Male_Negative', 'Female_Positive', 'Male_Positive']
TRAIN_RAW_DATA_FOLDER = 'data/raw/train/'
TEST_RAW_DATA_FOLDER = 'data/raw/test/'
PROCESSED_DATA_FOLDER = 'data/processed/'
SAVED_MODEL_FOLDER = 'result/weights/'
SAVED_LOGS_FOLDER = 'result/logs/'
SAVED_RESULTS_FOLDER = 'result/predictions/'

TO_SAVE = True
VERBOSE = 1
TEST_RATIO = 0.01
BATCH_SIZE = 1
EPOCH = 100

IMAGE_DATA_TYPE = np.int16
IMG_ROWS = 256
IMG_COLS = 256
IMG_CHNS = 1
IMG_DEPTH = 16

SMOOTH = 1.
BINARY_THR = 0.3

from src.utils.utility_functions import check_exist_folder
# check_exist_folder(TRAIN_RAW_DATA_FOLDER, create_if_not_exist=True)
# check_exist_folder(TEST_RAW_DATA_FOLDER, create_if_not_exist=True)
# check_exist_folder(PROCESSED_DATA_FOLDER, create_if_not_exist=True)
check_exist_folder(SAVED_MODEL_FOLDER, create_if_not_exist=True)
check_exist_folder(SAVED_LOGS_FOLDER, create_if_not_exist=True)
check_exist_folder(SAVED_RESULTS_FOLDER, create_if_not_exist=True)
