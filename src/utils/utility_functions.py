import numpy as np
import os
import shutil
import cv2
import pandas as pd
from sklearn.pipeline import Pipeline


class SelectColumnsTransformer:
    def __init__(self, columns=None):
        self.columns = [cl.lower() for cl in columns]

    def transform(self, X, **transform_params):
        columns = [c for c in X.columns if c.lower() in self.columns]
        return X[columns].copy()

    def fit(self, X, y=None, **fit_params):
        return self


def check_exist_folder(folder_path, create_if_not_exist=True):
    is_exist = os.path.exists(folder_path)
    if not is_exist and create_if_not_exist:
        os.makedirs(folder_path)
        return create_if_not_exist
    return is_exist


def delete_folder(folder_address):
    folder_address_abs = os.path.abspath(folder_address)
    shutil.rmtree(folder_address_abs, ignore_errors=True)
    # logger.info('local folder %s is deleted.' % folder_address_abs)
