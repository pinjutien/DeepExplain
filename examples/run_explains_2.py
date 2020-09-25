from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tempfile, sys, os
sys.path.insert(0, os.path.abspath('..'))
from deepexplain.tensorflow import DeepExplain
from explain_helper import explain_model# , load_image
from distutils.version import LooseVersion
import pandas as pd
import PIL
import tensorflow as tf
import numpy as np
from utils import kernel_density
from utils import plot, plt
from glob import glob
import shutil
import scipy.misc
from scipy import ndimage
from copy import deepcopy

def summary_plot(nrows, ncols, output, imag, base_imag, output_path):
    # y_min, y_max = 0, 13
    xs = imag
    attributions_ig = output["intgrad"]
    attributions_ig_base_line = output["intgrad_base"]
    attributions_dl = output["deeplift"]
    attributions_dl_base_line = output["deeplift_base"]
    attributions_exp_ig = output["expected_intgrad"]
    attributions_occ = output["occlusion"]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.reshape(nrows, ncols)
    for i in range(nrows):
        a1 = attributions_ig[i]
        a2 = attributions_ig_base_line[i]
        a3 = attributions_dl[i]
        a4 = attributions_dl_base_line[i]
        a5 = attributions_exp_ig[i]
        a6 = attributions_occ[i]        
        (h,w,c) = imag[i].shape
        if c == 1:
            axes[i, 0].imshow(np.squeeze(imag[i])) # .title('Original')
            axes[i, 1].imshow(np.squeeze(base_imag[i])) # .set_title('transformed')
        else:
            axes[i, 0].imshow(imag[i])# .title('Original')
            axes[i, 1].imshow(base_imag[i])# .set_title('transformed')
        plot(a1, axis=axes[i,2]).set_title('intgrad')
        plot(a2, axis=axes[i,3]).set_title('intgrad_base')
        plot(a3, axis=axes[i,4]).set_title('deeplift')
        plot(a4, axis=axes[i, 5]).set_title('deeplift_base')
        plot(a5, axis=axes[i, 6]).set_title('expected_ig')
        plot(a6, axis=axes[i, 7]).set_title('occlusion')        
        # kernel_arr += [log_dens]
        # log_dens = kernel_arr[i]
        # axes[i, 6].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
        # axes[i, 6].text(0, 5, "Gaussian Kernel Density")
        # axes[i, 6].set_ylim([y_min,y_max])
    
    # fig.savefig("apple2orange_explanation_map_1.jpg")
    fig.savefig(output_path)


def load_image_v1(baseline_target, base_image_path, all_images,
                  target_size=(256, 256, 3),
                  normalized_factor=255.0, baseline_type="gan"):
    # assert len(labels) == len(file_name), "length of labels and file_name should be the same."
    image_array = []
    (h, w, c) = target_size
    print(f"target_size: {target_size}")
    base_image_array = []
    prefix_file_array = []
    for image_f in all_images:
        file_name = os.path.basename(image_f)
        prefix_file_name = os.path.basename(file_name).split(".")[0]
        if baseline_type == "gan":
            baseline_f = prefix_file_name + f"_transformed_{baseline_target}.png"
        elif baseline_type == "closet":
            baseline_f = prefix_file_name + f"_closest_{baseline_target}.png"            
        base_path = os.path.join(base_image_path, baseline_f)
        if not os.path.exists(base_path):
            continue
        if c == 1:
            imag_temp = tf.keras.preprocessing.image.load_img(image_f, target_size=(h, w), color_mode="grayscale")
            base_imag_temp = tf.keras.preprocessing.image.load_img(base_path, target_size=(h,w), color_mode="grayscale")
        else:
            imag_temp = tf.keras.preprocessing.image.load_img(image_f, target_size=(h, w))
            base_imag_temp = tf.keras.preprocessing.image.load_img(base_path, target_size=(h,w))
        input_np= tf.keras.preprocessing.image.img_to_array(imag_temp)/normalized_factor
        base_input_np = tf.keras.preprocessing.image.img_to_array(base_imag_temp)/normalized_factor
        image_array += [input_np]
        base_image_array += [base_input_np]
        prefix_file_array += [prefix_file_name]
    imag = np.array(image_array, dtype=np.float)
    base_imag = np.array(base_image_array, dtype=np.float)
    return imag, base_imag, prefix_file_array


def process_base_image(original_image, to_process_image, noise_type, noise_scale):
    new_base_imag = []
    num_imag = original_image.shape[0]
    for i in range(num_imag):
        if "blur" in noise_type:
            # blur to_process_image as baseline
            base_imag_temp = ndimage.gaussian_filter(to_process_image[i], sigma=noise_scale)
        elif "uniform" in noise_type:
            # create uniform noise as baseline
            (h_, w_, c_) = to_process_image[i].shape
            base_imag_temp = np.random.uniform(0, 1.0, int(h_ * w_ *c_)).reshape(h_, w_, c_)
        elif "gaussian" in noise_type:
            # add gaussian noise to to_process_image as baseline.
            base_imag_temp = to_process_image[i]
            (h_, w_, c_) = base_imag_temp.shape
            base_imag_temp += np.random.normal(0, noise_scale, int(h_ * w_ * c_)).reshape(h_, w_, c_)
        else:
            raise Exception(f"unknow type: {noise_type}")
        new_base_imag += [base_imag_temp]
    new_base_imag = np.array(new_base_imag)
    return new_base_imag


def process_one_label(explain_types, model_path, target_label, root_image_path, num_class,
                      target_size, normalized_factor, noise_type=None, noise_scale=None, baseline_type = "gan"):
    # conside all the image given a label.
    # then use the corresponding baseline image.
    # target_label = 0
    # e.g. image_path = /home/ptien/from_ss/a2o/0/
    # baseline_folders = "transform_to_1"
    # explain_types = ["expected_intgrad", "intgrad", "intgrad_base", "deeplift", "deeplift_base"]
    image_path = os.path.join(root_image_path, target_label)
    print(f"imgae path: {image_path}")
    # output_path = os.path.join(image_path, "output")
    all_images = sorted(glob(os.path.join(image_path, "*.png")))
    # total_images = len(all_images)
    transformed_target = [ i for i in range(10)]
    output_path_ = os.path.join(image_path, "output", baseline_type)
    for baseline_target in transformed_target:
        trans_f = os.path.join(root_image_path, "baseline", baseline_type)
        output_path = os.path.join(output_path_,
                                   f"baseline_{baseline_target}_noise_type_{noise_type}_noise_scale_{noise_scale}")

        # prepare output folder:
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
            
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            plot_path = os.path.join(output_path, "plots")
            plot_path_numpy = os.path.join(output_path, "plots", "numpy")
            os.makedirs(plot_path)
            os.makedirs(plot_path_numpy)
            np_path = os.path.join(output_path, "explains", "numpy")
            for exp in explain_types:
                exp_path = os.path.join(np_path, exp)
                os.makedirs(exp_path)
            
        # load image and baseline
        print(f"transform folder: {trans_f}")
        imag, base_imag, prefix_file_names = load_image_v1(baseline_target, trans_f,
                                                           all_images,
                                                           target_size,
                                                           normalized_factor, baseline_type)
        if len(base_imag) == 0:
            shutil.rmtree(output_path)
            continue
            
        num_imag = imag.shape[0]
        print(f"Noise type: {noise_type}")
        if noise_type is None:
            pass
        elif noise_type == "blur_0":
            # blur original image as baseline
            del base_imag
            base_imag = process_base_image(imag, deepcopy(imag), noise_type, noise_scale)
        elif noise_type == "blur_1":
            # blur previous baseline
            base_imag = process_base_image(imag, deepcopy(base_imag), noise_type, noise_scale)            
        elif noise_type == "uniform":
            # blur original image as baseline            
            base_imag = process_base_image(imag, deepcopy(imag), noise_type, None)
        elif noise_type == "gaussian_0":
            # blur original image as baseline            
            base_imag = process_base_image(imag, deepcopy(imag), noise_type, noise_scale)            
        elif noise_type == "gaussian_1":
            # blur original image as baseline            
            base_imag = process_base_image(imag, deepcopy(base_imag), noise_type, noise_scale)
        else:
            raise Exception(f"unknow noise_type: {noise_type}")

        print(f"number of image: {num_imag}")
        # b_label = int(os.path.basename(os.path.normpath(trans_f)).replace("transform_to_", ""))
        base_target_label = [baseline_target]* num_imag
        y_label = [int(target_label)] * num_imag
        y_label_array = [y_label, base_target_label]
        # explain_dict = explain_model(model_path,
        #                              imag,
        #                              y_label,
        #                              num_class,
        #                              base_imag,
        #                              explain_types,
        #                              steps=200,
        #                              stochastic_mask_flag=False)
        explain_dict_arr = []
        for y_ in y_label_array:
            explain_dict_ = explain_model(model_path,
                                         imag,
                                         y_,
                                         num_class,
                                         base_imag,
                                         explain_types,
                                         steps=200,
                                         stochastic_mask_flag=False)
            explain_dict_arr += [explain_dict_]

        explain_dict_target = explain_dict_arr[0]
        explain_dict_baseline = explain_dict_arr[1]
        # explain_dict = {
        #     key: explain_dict_target.get(key) - explain_dict_baseline.get(key)
        #     for key in explain_dict_target.keys()
        # }
        explain_dict = {}
        for key in explain_dict_target.keys():
            # ["expected_intgrad", "intgrad", "intgrad_base", "deeplift", "deeplift_base", "occlusion"]
            if key in ["intgrad", "intgrad_base", "deeplift", "deeplift_base", "occlusion"]:
                explain_dict[key] = explain_dict_target.get(key) - explain_dict_baseline.get(key)
            elif key in ["expected_intgrad"]:
                explain_dict[key] = explain_dict_baseline.get(key)
            else:
                explain_dict[key] = explain_dict_target.get(key)
        nrows = num_imag
        kernel_plot = 0
        ncols = 2 + len(explain_types)+kernel_plot # 6
        output_path_f = os.path.join(plot_path,
                                     f"explains_summary_nt_{noise_type}_ns_{noise_scale}.jpg")
        # output plot
        summary_plot(nrows, ncols, explain_dict, imag, base_imag, output_path_f)

        # output input/baseline numpy array:
        for k in range(num_imag):
            f_input = prefix_file_names[k] + f"_original_nt_{noise_type}_ns_{noise_scale}.npy"
            f_baseline = prefix_file_names[k] + f"_baseline_nt_{noise_type}_ns_{noise_scale}.npy"
            f_in_path = os.path.join(plot_path, "numpy", f_input)
            f_b_path = os.path.join(plot_path, "numpy", f_baseline)
            f_in_arr = imag[k]
            f_b_arr = base_imag[k]
            np.save(f_in_path, f_in_arr)
            np.save(f_b_path, f_b_arr)
            
        # output explain numpy array
        for k in range(len(explain_dict_arr)):
            explain_dict_ = explain_dict_arr[k]
            for exp in explain_dict_.keys():
                exp_values = explain_dict_[exp]
                exp_path = os.path.join(np_path, exp)
                num_plot = exp_values.shape[0]
                assert num_imag == num_plot
                for i in range(num_plot):
                    f = prefix_file_names[i] + f"_nt_{noise_type}_ns_{noise_scale}_{k}.npy"
                    npy_path = os.path.join(exp_path, f)
                    plt_np_arr = exp_values[i]
                    np.save(npy_path, plt_np_arr)
            
        # for exp in explain_dict.keys():
        #     exp_values = explain_dict[exp]
        #     exp_path = os.path.join(np_path, exp)
        #     num_plot = exp_values.shape[0]
        #     assert num_imag == num_plot
        #     for i in range(num_plot):
        #         f = prefix_file_names[i] + f"_nt_{noise_type}_ns_{noise_scale}.npy"
        #         npy_path = os.path.join(exp_path, f)
        #         plt_np_arr = exp_values[i]
        #         np.save(npy_path, plt_np_arr)

    

if __name__ == '__main__':
    # root_image_path = "/home/ptien/from_ss/a2o/"
    # model_path = "/home/ptien/from_ss/a2o/model_keras.h5"
    # root_image_path = "/home/ptien/from_ss/mnist1_rs/"
    # root_image_path = "/home/ptien/from_ss/sanity_check/"
    # model_path = "/home/ptien/from_ss/mnist1/model_mnist1_keras.h5"
    model_path = "/home/ptien/from_ss/shap2/mnist/model_mnist_keras.h5"
    root_image_path = "/home/ptien/from_ss/2020-09-20/mnist/"
    explain_types = ["expected_intgrad", "intgrad", "intgrad_base", "deeplift", "deeplift_base", "occlusion"]
    # explain_types = ["expected_intgrad"]
    # explain_types = ["occlusion"]
    # target_size=[256, 256, 3]
    target_size=[28, 28, 1]
    baseline_type = "gan" # "closet" # 
    # "blur_0":"blur_1":"uniform":"gaussian_0":"gaussian_1":
    # noise_type = "blur_0" # None
    # blur: 5-50, gaussian: 0.5 -3
    # noise_scale = 5 # None
    noise_type_arr = [None, "blur_0", "blur_1", "uniform", "gaussian_0", "gaussian_1"]
    # noise_type_arr = [None]
    # noise_type_arr = [None]
    noise_scale_dict = {
        "blur": [0.1, 0.5, 1, 2],
        "gaussian": np.arange(0.1, 2, 0.5),
        "uniform": [None]
    }
    
    normalized_factor=255.0
    num_class = 10
    stochastic_mask_flag = False
    # all_classes = range(num_class)
    all_classes = [8]
    for noise_type in noise_type_arr:
        if noise_type is None:
            noise_scale_choice = [None]
        elif "blur" in noise_type:
            noise_scale_choice = noise_scale_dict["blur"]
        elif "gaussian" in noise_type:
            noise_scale_choice = noise_scale_dict["blur"]
        elif "uniform" in noise_type:
            noise_scale_choice = noise_scale_dict["uniform"]
        else:
            raise Exception(f"unknown noise type: {noise_type}")
        for noise_scale in noise_scale_choice:
            for i in all_classes:
                target_label = str(i)
                print(noise_type, noise_scale, target_label)
                process_one_label(explain_types, model_path, target_label, root_image_path, num_class,
                                  target_size, normalized_factor, noise_type, noise_scale, baseline_type)
            
    # # all_classes = [1, 4, 7]
    # for i in all_classes:
    #     # target_label = "0"
    #     target_label = str(i)
    #     print(target_label)
    #     process_one_label(model_path, target_label, root_image_path, num_class,
    #                       target_size, normalized_factor, noise_type, noise_scale)
