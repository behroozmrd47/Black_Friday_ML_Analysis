import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
# from skimage.measure import label
from collections import OrderedDict
# from skimage.segmentation import boundaries

import src.Constants as Cns

def plot_predictions(x, y , p):
    fig, axs = plt.subplots(1, 2)
    # mid_label_reg = label(mid_label)
    # boundary = boundaries.find_boundaries(mid_label_reg)
    xy = x * (-1 * (y - 1))
    axs[0].imshow(xy)
    axs[0].set_title('Image')
    axs[0].get_xaxis().set_visible(False)

    axs[1].imshow(p)
    axs[1].set_title('Label')
    axs[1].get_xaxis().set_visible(False)

    plt.show()
