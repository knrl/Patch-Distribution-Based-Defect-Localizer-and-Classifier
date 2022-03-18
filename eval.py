#
#   @author: Mehmet Kaan Erol
#
import os
import matplotlib
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries
from sklearn.metrics import precision_recall_curve

from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split

from classification_model.classifier import XClassifier

# configuration dictionary
from config import config_dict

def evaluate(scores, y_test, classifier_model_filepath, load, size):
    scores = scores.reshape(scores.shape[0], size * size)

    xgb = XClassifier(classifier_model_filepath=classifier_model_filepath, load=load)
    if (not load):
        X_train, scores, y_train, y_test = train_test_split(scores, y_test, test_size=0.2, random_state=0)
        xgb.train(X_train, y_train, save=True)
    y_pred = xgb.test(scores)

    print('y_test ',y_test, '\ny_pred ', y_pred)
    print('True Samples: ', np.sum(y_test == y_pred), " Num of Samples: ", len(y_test), " ", np.sum(y_test == y_pred)/len(y_test))

def plot_fig(test_img, scores, save_dir, class_name):
    num = len(scores)  
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        threshold  = threshold_otsu(scores[i]) * 1.25
        img = test_img[i]
        img = denormalization(img)
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 4, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')

        ax = ax_img[1].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[1].imshow(img, cmap='gray', interpolation='none')
        ax_img[1].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[1].title.set_text('Predicted heat map')

        ax_img[2].imshow(mask, cmap='gray')
        ax_img[2].title.set_text('Predicted mask, th ' + str(threshold))

        ax_img[3].imshow(vis_img)
        ax_img[3].title.set_text('Result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)
        fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()

def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x

if (__name__ == '__main__'):
    import test
    from config import config_dict
    test_img, scores, y_true = test.run(
            test_dataset_path = config_dict['test_dataset_path'], 
            class_name = config_dict['class_name'], 
            test_batch_size = config_dict['test_batch_size'],
            distance_metric=config_dict['distance_metric'],
            train_feature_filepath = config_dict['train_feature_filepath'],
            is_single = config_dict['is_single'],
            size = config_dict['size']
        )

    size = config_dict['size']
    class_name = config_dict['class_name']
    save_dir = config_dict['save_dir']
    classifier_model_filepath = config_dict['classifier_model_filepath']
    load = config_dict['load']
    os.makedirs(save_dir, exist_ok=True)

    evaluate(scores, y_true, classifier_model_filepath, load, size)
    plot_fig(test_img, scores, save_dir, class_name)
