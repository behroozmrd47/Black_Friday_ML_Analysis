RUN_NAME = 'FF_All_Nov19'

TRAIN_RAW_DATA_FOLDER = 'data/train.csv'
TEST_RAW_DATA_FOLDER = 'data/test.csv'

SAVED_MODEL_FOLDER = 'result/weights/'
SAVED_LOGS_FOLDER = 'result/logs/'
SAVED_RESULTS_FOLDER = 'result/predictions/'

TO_SAVE = True
VERBOSE = 1
TEST_RATIO = 0.01
BATCH_SIZE = 1
EPOCH = 100

from src.utils.utility_functions import check_exist_folder
check_exist_folder(SAVED_MODEL_FOLDER, create_if_not_exist=True)
check_exist_folder(SAVED_LOGS_FOLDER, create_if_not_exist=True)
check_exist_folder(SAVED_RESULTS_FOLDER, create_if_not_exist=True)
