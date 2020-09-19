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
    image_path = "/home/ptien/tfds-download/apple2orange/testA/"
    output_path = "/home/ptien/tfds-download/out2/model.ckpt-74000/"
    base_image_path = "/home/ptien/tfds-download/apple2orange/stargan/model.ckpt-74000/"
    model_path = "/home/ptien/tfds-download/models_ckpts/classification/a2o/apple2orange.h5"
    explain_types = ["intgrad", "intgrad_base", "deeplift", "deeplift_base"]
    target_size=[256, 256, 3]
    normalized_factor=255.0
    num_class = 2
    stochastic_mask_flag = False    
    all_images = glob(image_path + "*.jpg")
    c = 0
    batch_size = 10
    total_images = len(all_images)
    df_file = []
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        plot_path = os.path.join(output_path, "plots")
        np_path = os.path.join(output_path, "explains", "numpy")
        os.makedirs(plot_path)
        for exp in explain_types:
            exp_path = os.path.join(np_path, exp)
            os.makedirs(exp_path)
    
    while len(all_images) > 0:
        print(c, len(all_images))
        batch_images = all_images[:batch_size]
        file_name = [ f.split("/")[-1] for f in batch_images]
        df_temp = pd.DataFrame(data=file_name, columns=["file_name"])
        df_temp["batch"] = c
        df_file += [df_temp]
        # file_name = ["n07740461_10011.jpg", "n07740461_240.jpg", "n07740461_14960.jpg", 
        #              "n07740461_2770.jpg", "n07740461_14600.jpg", "n07740461_14531.jpg", 
        #              "n07740461_14300.jpg", "n07740461_10371.jpg", "n07740461_10571.jpg"]
        labels = [0]*len(file_name) # [0,0,0,0,0,0]
        imag, y_label, base_imag = load_image(image_path, base_image_path, file_name, labels, num_class,
                                              target_size, normalized_factor)
        kernel_arr, local_min_max, X_plot = kernel_density(imag, base_imag, file_name,
                                                           bandwidth =-1, op="min",
                                                           filter_=False, custom_std=False, iqr_choice=True)
        output = explain_model(model_path, imag, y_label, num_class, base_imag, explain_types,
                               steps=200,
                               stochastic_mask_flag=stochastic_mask_flag)

        nrows = len(file_name)
        kernel_plot = 1
        ncols = 2 + len(explain_types)+kernel_plot # 6
        output_path_f = os.path.join(plot_path, "explains_summary_{x}.jpg".format(x=c))
        summary_plot(nrows, ncols, output, imag, base_imag, output_path_f)
        # output explain numpy array
        for exp in output.keys():
            exp_values = output[exp]
            exp_path = os.path.join(np_path, exp)
            num_plot = exp_values.shape[0]
            for i in range(num_plot):
                f = file_name[i] + ".npy"
                npy_path = os.path.join(exp_path, f)
                plt_np_arr = exp_values[i]
                np.save(npy_path, plt_np_arr)
        all_images = all_images[batch_size:]
        c += 1
        del output
    pd.concat(df_file).to_csv(output_path + "filename.csv")
