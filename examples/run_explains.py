from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tempfile, sys, os
sys.path.insert(0, os.path.abspath('..'))
from deepexplain.tensorflow import DeepExplain
from explain_helper import explain_model, load_image
from distutils.version import LooseVersion
import pandas as pd
import PIL
import tensorflow as tf
import numpy as np
from utils import kernel_density
from utils import plot, plt
from glob import glob



def summary_plot(nrows, ncols, output, imag, base_imag, output_path):
    y_min, y_max = 0, 13
    xs = imag
    attributions_ig = output["intgrad"]
    attributions_ig_base_line = output["intgrad_base"]
    attributions_dl = output["deeplift"]
    attributions_dl_base_line = output["deeplift_base"]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
    for i in range(nrows):
        a1 = attributions_ig[i]
        a2 = attributions_ig_base_line[i]
        a3 = attributions_dl[i]
        a4 = attributions_dl_base_line[i]
        axes[i, 0].imshow(imag[i])# .title('Original')
        axes[i, 1].imshow(base_imag[i])# .set_title('transformed')
        plot(a1, axis=axes[i,2]).set_title('intgrad')
        plot(a2, axis=axes[i,3]).set_title('intgrad_base')
        plot(a3, axis=axes[i,4]).set_title('deeplift')
        plot(a4, axis=axes[i, 5]).set_title('deeplift_base')
        # kernel_arr += [log_dens]
        log_dens = kernel_arr[i]
        axes[i, 6].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
        axes[i, 6].text(0, 5, "Gaussian Kernel Density")
        axes[i, 6].set_ylim([y_min,y_max])
    
    # fig.savefig("apple2orange_explanation_map_1.jpg")
    fig.savefig(output_path)
    

if __name__ == '__main__':
    image_path = "/Users/pin-jutien/tfds-download/apple2orange/testA/"
    output_path = "/Users/pin-jutien/tfds-download/apple2orange/out/"
    # base_image_path = "/Users/pin-jutien/tfds-download/apple2orange/experiment2-500000/generated_y/"
    # base_image_path = "/Users/pin-jutien/tfds-download/apple2orange/experiment-w50/generated_y/"
    base_image_path = "/Users/pin-jutien/tfds-download/apple2orange/stargan/model.ckpt-120000/"
    model_path = "/Users/pin-jutien/tfds-download/models_ckpts/classification/a2o/apple2orange.h5"
    explain_types = ["intgrad", "intgrad_base", "deeplift", "deeplift_base"]
    num_class = 2
    stochastic_mask_flag = False    
    all_images = glob(image_path + "*.jpg")
    c = 0
    batch_size = 10
    total_images = len(all_images)
    
    while len(all_images) > 0:
        print(c)
        batch_images = all_images[:batch_size]
        file_name = [ f.split("/")[-1] for f in batch_images]
        # file_name = ["n07740461_10011.jpg", "n07740461_240.jpg", "n07740461_14960.jpg", 
        #              "n07740461_2770.jpg", "n07740461_14600.jpg", "n07740461_14531.jpg", 
        #              "n07740461_14300.jpg", "n07740461_10371.jpg", "n07740461_10571.jpg"]
        labels = [0]*len(file_name) # [0,0,0,0,0,0]
        imag, y_label, base_imag = load_image(image_path, base_image_path, file_name, labels, num_class)
        kernel_arr, local_min_max, X_plot = kernel_density(imag, base_imag, file_name,
                                                           bandwidth =-1, op="min",
                                                           filter_=False, custom_std=False, iqr_choice=True)
        output = explain_model(model_path, imag, y_label, num_class, base_imag, explain_types,
                               steps=200,
                               stochastic_mask_flag=stochastic_mask_flag)

        nrows = len(file_name)
        kernel_plot = 1
        ncols = 2 + len(explain_types)+kernel_plot # 6
        output_path_f = os.path.join(output_path, "explains_summary_{x}.jpg".format(x=c))
        summary_plot(nrows, ncols, output, imag, base_imag, output_path_f)
        all_images = all_images[batch_size:]
        c += 1
        del output
