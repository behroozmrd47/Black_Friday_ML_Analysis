import os
import keras
import matplotlib.pyplot as plt
import src.Constants as Cns


class PlotLearning(keras.callbacks.Callback):
    """
    Callback function for plotting the training results at the end of each epoch.
    """
    def __init__(self, train_object):
        self.training = train_object
        super(PlotLearning, self).__init__()

    def on_train_begin(self, logs={}):
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.save_adrs = os.path.join(Cns.SAVED_LOGS_FOLDER, self.training.RUN_ID + 'train_chart.png')

    def on_epoch_end(self, epoch, logs={}):
        self.x.append(epoch)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('dice3d_metric'))
        self.val_acc.append(logs.get('val_dice3d_metric'))

        if epoch % 5 == 1:
            fig, ax1 = plt.subplots(figsize=(10, 5))
            lns1 = ax1.plot(self.x, self.acc, label="Train Dice Score", color='orange', linewidth=2.)
            lns2 = ax1.plot(self.x, self.val_acc, label="Test Dice Score", color='blue', linewidth=2.)
            ax1.set_ylabel('Dice Score', fontsize=15)
            ax1.set_xlabel('Epoch', fontsize=15)
            ax1.set_title(self.training.RUN_ID)
            ax2 = ax1.twinx()
            lns3 = ax2.plot(self.x, self.losses, '--', label="Train Loss", color='orange')
            lns4 = ax2.plot(self.x, self.val_losses, '--', label="Test Loss", color='blue')
            ax2.set_ylabel('Loss', fontsize=15)
            lns = lns1 + lns2 + lns3 + lns4
            ax1.legend(lns, [l.get_label() for l in lns])
            ax1.grid(True)
            plt.xlim([-0.05, epoch + .05])
            plt.savefig(self.save_adrs)
