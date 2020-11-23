import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

import src.Constants as Cns
from src.utils.utility_functions import de_stack_images, binarize_images
from src.model_dev.Callback import PlotLearning
from src.model_dev.Process_Raw_Input import process_concat_images_3d
from src.models.loss.loss import dice3d_metric, bce_dice_loss

from src.models.Unet3D_res_rnn import unet3d_res_rnn as model_func
# from src.models.Unet3D_res import unet3d_res as model_func

tf.compat.v1.disable_eager_execution()
np.random.seed(144)
tf.random.set_seed(144)
K.set_image_data_format('channels_last')


class TPE3D:
    """
    Class for Training, Predicting and Evaluating and hence TPE class.
    """
    RUN_ID: str

    def __init__(self, model_func, x_train, y_train, x_test, y_test, **kwargs):
        """
        Constants for each training class object.
        :param model_func: the model class
        :param x_train: train image dataset
        :param y_train: train mask dataset
        :param x_test: test image dataset
        :param y_test: test mask dataset
        :param kwargs: etc arguments
        """
        kwargs = {k.upper(): v for k, v in kwargs.items()}
        self.RUN_NAME = kwargs.get('RUN_NAME', Cns.RUN_NAME_3D)
        self.TO_SAVE = kwargs.get('TO_SAVE', Cns.TO_SAVE)
        self.VERBOSE = kwargs.get('VERBOSE', Cns.VERBOSE)

        self.IMAGE_DATA_TYPE = np.int16
        self.IMG_ROWS = kwargs.get('IMG_ROWS', Cns.IMG_ROWS)
        self.IMG_COLS = kwargs.get('IMG_COLS', Cns.IMG_COLS)
        self.IMG_CHNS = kwargs.get('IMG_CHNS', Cns.IMG_CHNS)
        self.IMG_DEPTH = kwargs.get('IMG_DEPTH', Cns.IMG_DEPTH)

        self.TEST_RATIO = kwargs.get('TEST_RATIO', Cns.TEST_RATIO)
        self.BATCH_SIZE = kwargs.get('BATCH_SIZE', Cns.BATCH_SIZE)
        self.EPOCH = kwargs.get('EPOCH', Cns.EPOCH)

        # self.IMAGE_RAW_DATA_FOLDER =  kwargs.get('TEST_RATIO', Cns.TEST_RATIO)'data/label/image/'
        # self.LABEL_RAW_DATA_FOLDER = 'data/label/label/'
        # self.SAVED_MODEL_FOLDER = 'model/weight/'
        # self.SAVED_LOGS_FOLDER = 'model/log/'
        # self.SAVED_RESULTS_FOLDER = 'model/predictions/'
        # check_exist_folder(self.SAVED_MODEL_FOLDER, create_if_not_exist=True)
        # check_exist_folder(self.SAVED_LOGS_FOLDER, create_if_not_exist=True)
        # check_exist_folder(self.SAVED_RESULTS_FOLDER, create_if_not_exist=True)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = None
        self.model_func = model_func
        self.model = self.model_func(img_rows=self.IMG_ROWS, img_cols=self.IMG_COLS, img_chns=self.IMG_CHNS,
                                     img_depth=self.IMG_DEPTH)

        self.optimizer = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199, name='AdamLR5')
        self.loss = bce_dice_loss
        self.loss_name = 'bce-dice'
        self.metrics = [dice3d_metric]
        self.make_run_id()

    def make_run_id(self):
        self.RUN_ID = '%s_%s_%s_EP%d_BS%d_%dx%d' % (self.RUN_NAME, self.model.name, self.loss_name, self.EPOCH,
                                                    self.BATCH_SIZE, self.IMG_ROWS, self.IMG_COLS)

    def set_img_size(self, img_rows, img_cols):
        self.IMG_ROWS = img_rows
        self.IMG_COLS = img_cols
        self.model = self.model_func(img_rows=self.IMG_ROWS, img_cols=self.IMG_COLS, img_chns=self.IMG_CHNS)
        self.make_run_id()
        return self

    def set_model(self, model_class):
        self.model = model_class(img_rows=self.IMG_ROWS, img_cols=self.IMG_COLS, img_chns=self.IMG_CHNS)
        self.make_run_id()
        return self

    def set_run_name(self, run_name):
        self.RUN_NAME = run_name
        self.make_run_id()
        return self

    def set_num_epochs(self, new_epochs):
        self.EPOCH = new_epochs
        self.make_run_id()
        return self

    def set_batch_size(self, new_batch_size):
        self.BATCH_SIZE = new_batch_size
        self.make_run_id()
        return self

    def set_optimizer(self, optimizer_inp):
        self.optimizer = optimizer_inp
        return self

    def set_loss(self, loss, loss_name):
        self.loss = loss
        self.loss_name = loss_name
        self.make_run_id()
        return self

    def set_metrics(self, metrics_inp):
        self.metrics = metrics_inp
        return self

    def train(self, **kwargs):
        """
        Train the model and save if to_save is True.
        :return: train results dictionary
        """
        run_id = kwargs.get('run_id', self.RUN_ID)
        to_save = kwargs.get('to_save', self.TO_SAVE)
        x_train = kwargs.get('x_train', self.x_train)
        y_train = kwargs.get('y_train', self.y_train)
        x_test = kwargs.get('x_test', self.x_test)
        y_test = kwargs.get('y_test', self.y_test)

        print('_' * 70)
        print('\nTrain process started on train%s/test%s.' % (str(x_train.shape), str(x_test.shape)))
        print('\nCreating and compiling model %s' % self.model.name)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        callbacks = self.get_callbacks()

        if x_test is not None:
            train_history = self.model.fit(x_train, y_train, batch_size=self.BATCH_SIZE, epochs=self.EPOCH,
                                           verbose=self.VERBOSE, shuffle=False, validation_data=(x_test, y_test),
                                           callbacks=callbacks)
        else:
            train_history = self.model.fit(x_train, y_train, batch_size=self.BATCH_SIZE, epochs=self.EPOCH,
                                           verbose=self.VERBOSE, shuffle=False, validation_split=self.TEST_RATIO,
                                           callbacks=callbacks)
        self.train_hist_df = pd.DataFrame(train_history.history, index=train_history.epoch)
        print('\nTraining finished')

        if to_save:
            if not os.path.exists(Cns.SAVED_MODEL_FOLDER):
                os.mkdir(Cns.SAVED_MODEL_FOLDER)
            model_weights_addr = os.path.join(Cns.SAVED_MODEL_FOLDER, run_id + '_final_weights.h5')
            self.model.save_weights(model_weights_addr)
            print('Final & best weights saved at %s.' % model_weights_addr)

            train_history_addr = os.path.join(Cns.SAVED_LOGS_FOLDER, run_id + 'train_history.csv')
            self.train_hist_df.to_csv(path_or_buf=train_history_addr)
            print('Train history saved at %s.' % train_history_addr)
        self.train_res_dic = {'model': self.model, 'train_history': train_history}
        return self.train_res_dic

    def predict(self, x_test=None, y_test=None, **kwargs):
        """
        Predict the model and save predication results if to_save is True.
        :param x_test: test image dataset
        :param y_test: test mask dataset
        :param kwargs: etc arguments
        :return: predictions results dictionary
        """
        print('_' * 70)
        kwargs = {k.upper(): v for k, v in kwargs.items()}
        to_save = kwargs.get('TO_SAVE', self.TO_SAVE)
        run_id = kwargs.get('RUN_ID', self.RUN_ID)
        weight_path = kwargs.get('WEIGHT_PATH', None)

        if weight_path:
            self.model = self.model_func(img_rows=self.IMG_ROWS, img_cols=self.IMG_COLS, img_chns=self.IMG_CHNS,
                                         img_depth=self.IMG_DEPTH, pretrained_weight_path=weight_path)
            print("model loaded from %s" % weight_path)

        best_weights_path = os.path.join(Cns.SAVED_MODEL_FOLDER, self.RUN_ID + '_best_weights.h5')
        self.model_best = self.model_func(img_rows=self.IMG_ROWS, img_cols=self.IMG_COLS, img_chns=self.IMG_CHNS,
                                          img_depth=self.IMG_DEPTH, pretrained_weight_path=best_weights_path)
        print("model loaded from %s" % weight_path)

        if x_test is None or y_test is None:
            x_test = self.x_test
            y_test = self.y_test

        self.y_pred = self.model.predict(x_test, batch_size=self.BATCH_SIZE, verbose=self.VERBOSE)
        self.y_pred_best = self.model_best.predict(x_test, batch_size=self.BATCH_SIZE, verbose=self.VERBOSE)
        print('\nPrediction on test data shape %s output shape %s.' % (x_test.shape, self.y_pred.shape))

        if to_save:
            if not os.path.exists(Cns.SAVED_RESULTS_FOLDER):
                os.mkdir(Cns.SAVED_RESULTS_FOLDER)
            np.save(os.path.join(Cns.SAVED_RESULTS_FOLDER, run_id + '_test_feature.npy'), x_test)
            np.save(os.path.join(Cns.SAVED_RESULTS_FOLDER, run_id + '_test_label.npy'), y_test)
            np.save(os.path.join(Cns.SAVED_RESULTS_FOLDER, run_id + '_test_predict_final.npy'), self.y_pred)
            np.save(os.path.join(Cns.SAVED_RESULTS_FOLDER, run_id + '_test_predict_best.npy'), self.y_pred_best)
            print('Images, masks & predicted label saved to ' + os.path.join(Cns.SAVED_RESULTS_FOLDER,
                                                                             run_id + '_test_predict.npy'))
        self.pred_res_dic = {'x_test': x_test, 'y_test': y_test, 'y_pred': self.y_pred}
        return self.pred_res_dic

    def evaluate(self, y_test=None, y_pred=None):
        """
        Evaluate predictions results.
        :param y_test: test mask dataset
        :param y_pred: predictions mask data predicted on test data
        :param is_plot: to plot the results
        :return: evaluation results
        """
        if y_test is None or y_pred is None:
            y_test = self.y_test
            y_pred = self.y_pred

        y_test_de_stack = de_stack_images(y_test, stack_num=self.IMG_DEPTH)
        y_test_b = binarize_images(y_test_de_stack, threshold=Cns.BINARY_THR)

        y_pred_de_stack = de_stack_images(y_pred, stack_num=self.IMG_DEPTH)
        y_pred_b = binarize_images(y_pred_de_stack, threshold=Cns.BINARY_THR)

        y_pred_best_de_stack = de_stack_images(self.y_pred_best, stack_num=self.IMG_DEPTH)
        y_pred_best_b = binarize_images(y_pred_best_de_stack, threshold=Cns.BINARY_THR)

        dice_score = (2. * (y_test_b * y_pred_b).sum()) / (y_test_b.sum() + y_pred_b.sum())
        dice_score_best = (2. * (y_test_b * y_pred_best_b).sum()) / (y_test_b.sum() + y_pred_best_b.sum())

        print('Final dice score accuracy on test dataset is %f02' % dice_score)
        print('Best dice score accuracy on test dataset is %f02' % dice_score_best)
        return dice_score

    def get_callbacks(self):
        """
        Generates callbacks for training.
        :return: array of callbacks
        """
        model_checkpoint = ModelCheckpoint(os.path.join(Cns.SAVED_MODEL_FOLDER, self.RUN_ID + '_best_weights.h5'),
                                           monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True)
        # stop_train = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=0.000001)
        if not os.path.exists(Cns.SAVED_LOGS_FOLDER):
            os.mkdir(Cns.SAVED_LOGS_FOLDER)
        csv_logger = CSVLogger(os.path.join(Cns.SAVED_LOGS_FOLDER, self.RUN_ID + 'train_logs.txt'), separator=',',
                               append=False)
        plot_learning = PlotLearning(train_object=self)
        return [model_checkpoint, reduce_lr, csv_logger, plot_learning]

    def tpe_main(self, **kwargs):
        self.train(**kwargs)
        # weight_path = os.path.join(Cns.SAVED_MODEL_FOLDER, self.RUN_ID + '_best_weights.h5')
        self.predict(**kwargs)
        self.evaluate()
        # self.plot_train_hist()
        print(self.RUN_ID + '**Completed**')

    def plot_train_hist(self):
        """
        Plot train history at the end of training.
        :return: None
        """
        ax = self.train_hist_df.plot(y=['loss', 'val_loss', 'dice_metric', 'val_dice_metric'], grid=True,
                                     secondary_y=['dice_metric', 'val_dice_metric'], title=self.RUN_ID)
        ax.set_xlabel('Epoch #')
        ax.set_ylabel('Loss Value')
        ax.right_ax.set_ylabel('Dice Score')
        plt.show()


if __name__ == '__main__':
    print('Loading and pre_processing train data...')
    x_train, y_train = process_concat_images_3d(input_data_folder_adr=Cns.TRAIN_RAW_DATA_FOLDER,
                                                dtype=Cns.IMAGE_DATA_TYPE)
    x_test, y_test = process_concat_images_3d(input_data_folder_adr=Cns.TEST_RAW_DATA_FOLDER, dtype=Cns.IMAGE_DATA_TYPE)
    print('train dataset shape: %s' % str(x_train.shape))
    print('validation dataset shape: %s' % str(x_test.shape))
    print('-' * 70)

    tpe_3d_bce = TPE3D(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, model_func=model_func)
    tpe_3d_bce.tpe_main()
    # tpe_3d_bce.train()
    # weight_path = os.path.join(Cns.SAVED_MODEL_FOLDER, tpe_3d_bce.RUN_ID + '_best_weights.h5')
    # tpe_3d_bce.predict(weight_path=weight_path)
    # tpe_3d_bce.evaluate()
    # self.plot_train_hist()
    # print(tpe_3d_bce.RUN_ID + '**Completed**')
