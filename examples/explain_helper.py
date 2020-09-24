import pandas as pd
import PIL
import tensorflow as tf
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from scipy.stats import iqr
from utils import plot, plt
import glob
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from deepexplain.tensorflow import DeepExplain
from get_labels_data import get_baseline_data


def load_image(image_path, base_image_path, file_name, labels, num_class,
               target_size=(256, 256, 3),
               normalized_factor=255.0):
    assert len(labels) == len(file_name), "length of labels and file_name should be the same."
    image_array = []
    base_image_array = []
    for f in file_name:
        # imag_path =  "/Users/ptien/tfds-download/apple2orange/testA/" + f
        # base_path = "/Users/ptien/tfds-download/apple2orange/experiment2-500000/generated_y/generated_from_" + f
        imag_path = image_path + f
        base_path = base_image_path + "generated_from_" + f
        imag_temp = tf.keras.preprocessing.image.load_img(imag_path, target_size=target_size)
        base_imag_temp = tf.keras.preprocessing.image.load_img(base_path, target_size=target_size)
        input_np= tf.keras.preprocessing.image.img_to_array(imag_temp)/normalized_factor
        base_input_np = tf.keras.preprocessing.image.img_to_array(base_imag_temp)/normalized_factor
        image_array += [input_np]
        base_image_array += [base_input_np]
    imag = np.array(image_array, dtype=np.float)
    base_imag = np.array(base_image_array, dtype=np.float)
    y_label = labels # apple
    return imag, y_label, base_imag

def explain_model(model_path,
                  imag,
                  y_label,
                  num_class,
                  base_imag,
                  explain_types,
                  steps=100,
                  stochastic_mask_flag=False,
                  data_type="mnist"):
    tf.keras.backend.clear_session()
    # tf.reset_default_graph()
    model = tf.keras.models.load_model(model_path)
    with DeepExplain(session=tf.keras.backend.get_session()) as de:  # <-- init DeepExplain context
        # Need to reconstruct the graph in DeepExplain context, using the same weights.
        # With Keras this is very easy:
        # 1. Get the input tensor to the original model
        input_tensor = model.layers[0].input

        # 2. We now target the output of the last dense layer (pre-softmax)
        # To do so, create a new model sharing the same layers untill the last dense (index -2)
        # fModel = Model(inputs=input_tensor, outputs = model.layers[-2].output)
        fModel = tf.keras.models.Model(inputs=model.input, outputs = model.output)
        target_tensor = fModel(input_tensor) # fModel(input_tensor)

        # xs = x_test[0:10]
        # ys = y_test[0:10]
        xs = imag
        # ys = np.array([tf.keras.utils.to_categorical(y_label, 2)])
        ys = tf.keras.utils.to_categorical(y_label, num_class)
        output = {}
        for type_ in explain_types:
            print("process {x}".format(x=type_))
            if "_base" in type_:
                assert base_imag is not None, "Please provide non-trivial baseline.{x}".format(x=type_)
                attributions_base_ = de.explain(type_.split("_base")[0], target_tensor, input_tensor, xs, ys=ys,
                                                baseline=base_imag, steps=steps, stochastic_mask_flag=stochastic_mask_flag)
                output[type_] = attributions_base_
            elif type_ == "expected_intgrad":
                # import pdb; pdb.set_trace()
                base_for_exp_ig = get_baseline_data(data_type, y_label, subsample=steps)
                attributions_ = de.explain("intgrad", target_tensor, input_tensor, xs, ys=ys,
                                           baseline=base_for_exp_ig, steps=steps, stochastic_mask_flag=type_)
                output[type_] = attributions_
            elif type_ == "occlusion":
                attributions_ = de.explain(type_, target_tensor, input_tensor, xs, ys=ys)
                output[type_] = attributions_
            else:
                attributions_ = de.explain(type_, target_tensor, input_tensor, xs, ys=ys, baseline=None, steps=steps, stochastic_mask_flag=False)
                output[type_] = attributions_

        # if "grad*input" in explain_types:
        #     attributions_gradin = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys)
        #     output["grad*input"] = attributions_gradin
        # if "saliency" in explain_types:
        #     attributions_sal   = de.explain('saliency', target_tensor, input_tensor, xs, ys=ys)
        #     output["saliency"] = attributions_sal
        # if "intgrad" in explain_types:
        #     attributions_ig = de.explain('intgrad', target_tensor, input_tensor, xs, ys=ys)
        #     output["intgrad"] = attributions_ig
        # if "intgrad_base" in explain_types:
        #     assert base_imag != None, "Please provide non-trivial baseline."
        #     attributions_ig_base_line = de.explain('intgrad', target_tensor, input_tensor, xs, ys=ys,baseline=base_imag)
        #     output["intgrad_base"] = attributions_ig_base_line
        # if "deeplift" in explain_types:
        #     attributions_dl = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys)
        #     output["deeplift"] = attributions_dl
        # if "deeplift_base" in explain_types:
        #     attributions_dl_base_line = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys,baseline=base_imag)
        #     output["deeplift_base"] = attributions_dl_base_line
        # if "elrp" in explain_types:
        #     attributions_elrp  = de.explain('elrp', target_tensor, input_tensor, xs, ys=ys)
        #     output["elrp"] = attributions_elrp
        
        # attributions_gradin = de.explain('grad*input', target_tensor, input_tensor, xs, ys=ys)
        #attributions_sal   = de.explain('saliency', target_tensor, input_tensor, xs, ys=ys)
        # attributions_ig    = de.explain('intgrad', target_tensor, input_tensor, xs, ys=ys)
        # attributions_ig_base_line = de.explain('intgrad', target_tensor, input_tensor, xs, ys=ys,baseline=base_imag)
        # attributions_dl = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys)
        # attributions_dl_base_line = de.explain('deeplift', target_tensor, input_tensor, xs, ys=ys,baseline=base_imag)
        #attributions_elrp  = de.explain('elrp', target_tensor, input_tensor, xs, ys=ys)
        #attributions_occ   = de.explain('occlusion', target_tensor, input_tensor, xs, ys=ys)

        # Compare Gradient * Input with approximate Shapley Values
        # Note1: Shapley Value sampling with 100 samples per feature (78400 runs) takes a couple of minutes on a GPU.
        # Note2: 100 samples are not enough for convergence, the result might be affected by sampling variance
        # attributions_sv     = de.explain('shapley_sampling', target_tensor, input_tensor, xs, ys=ys, samples=100)
        # return attributions_ig, attributions_ig_base_line, attributions_dl, attributions_dl_base_line
        return output


def kernel_density(original_image, gan_image, file_name, bandwidth = 0.02, op="min", 
                   filter_=True, custom_std=False, iqr_choice=False):
    num_fig = original_image.shape[0]
    assert num_fig == len(file_name), "Number of figure is not the same."
    diff_image_base = original_image - gan_image
    X_plot = np.linspace(-1, 1, 1000)[:, np.newaxis]
    bins = np.linspace(-1, 1, 1000)
    local_min_max = {}
    comparison_op = {
        "min": np.less,
        "max": np.greater
    }
    print("compare operator: {x}".format(x=op))
    kernel_arr = []
    for i in range(num_fig):
        X = diff_image_base[i].reshape(-1,1)
        n = len(X)
        if custom_std:
            bandwidth = custom_std*X.std()
        if iqr_choice:
            # import pdb; pdb.set_trace()
            iqr_num = iqr(X)
            bandwidth = 0.9* min(X.std(), iqr_num/1.34)*pow(n, -0.2)
            
        if filter_:
            X = [ xx for xx in X if abs(xx) >= bandwidth]
        # Gaussian KDE
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
        log_dens = kde.score_samples(X_plot)
        kernel_arr += [log_dens]
        
        compare_op = comparison_op[op]
        kernel_y = np.exp(log_dens)
        local_indexs = argrelextrema(kernel_y, compare_op)[0]
        local_min_max[file_name[i]] = X_plot[local_indexs]
        del X
    return kernel_arr, local_min_max, X_plot


def load_batch_images(image_paths, file_name):
    res = []
    for p in image_paths:
        res_p = []
        for f in file_name:
            path_ = p + f
            imag_temp = tf.keras.preprocessing.image.load_img(path_, target_size=(256, 256, 3))
            input_np= tf.keras.preprocessing.image.img_to_array(imag_temp)/255.
            res_p += [input_np]
        res += [res_p]
    res = np.array(res)
    return res


def show_kd_plot(image_paths, file_name, filter_=False, output=None, custom_std=False, iqr_choice=False):
    images_collection = load_batch_images(image_paths, file_name)
    original_imag = images_collection[0]
    gan_imag = images_collection[1]
    kernel_arr, local_min_max, X_plot = kernel_density(original_imag, gan_imag, file_name, 
                                                       bandwidth = 0.02, op="min", filter_=filter_, 
                                                       custom_std=custom_std, iqr_choice=iqr_choice)
    gan_imag2 = images_collection[2]
    kernel_arr2, local_min_max2, X_plot2 = kernel_density(original_imag, gan_imag2, file_name, 
                                                          bandwidth = 0.02, op="min", filter_=filter_,
                                                         custom_std=custom_std, iqr_choice=iqr_choice)
    gan_imag3 = images_collection[3]
    kernel_arr3, local_min_max3, X_plot3 = kernel_density(original_imag, gan_imag3, file_name, 
                                                          bandwidth = 0.02, op="min", filter_=filter_,
                                                         custom_std=custom_std, iqr_choice=iqr_choice)
    nrows = len(file_name)
    ncols = 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.reshape(nrows, ncols)
    y_min,y_max = 0, 10
    for i in range(nrows):
        # a4 = attributions_dl_base_line[i]
        axes[i, 0].imshow(original_imag[i])# .set_title('Original')
        axes[i, 1].imshow(gan_imag[i])# .set_title('gan-1')
        axes[i, 2].imshow(gan_imag2[i])# .set_title('gan-2')
        axes[i, 3].imshow(gan_imag3[i])# .set_title('gan-3')
        log_dens = kernel_arr[i]
        log_dens2 = kernel_arr2[i]
        log_dens3 = kernel_arr3[i]
        axes[i, 4].fill(X_plot[:, 0], np.exp(log_dens), fc='green', alpha=0.5, label="gan-1")
        axes[i, 4].fill(X_plot2[:, 0], np.exp(log_dens2), fc='blue', alpha=0.3, label="gan-2")
        axes[i, 4].fill(X_plot3[:, 0], np.exp(log_dens3), fc='red', alpha=0.5, label="gan-3")
        axes[i, 4].axvline(0, y_min, y_max, c="gray", alpha=0.1)
        # axes[i, 0].text(0, 5, str(i))
        axes[i, 4].set_ylim([y_min,y_max])
    
    if output:
        fig.savefig(output)


if __name__ == '__main__':
    # image_path = "/Users/ptien/tfds-download/apple2orange/testA/"
    # gan_image_path_1 = "/Users/ptien/tfds-download/apple2orange/experiment2-500000/generated_y/generated_from_"
    # gan_image_path_2 = "/Users/ptien/tfds-download/apple2orange/experiment-1000/generated_y/generated_from_"
    # gan_image_path_3 = "/Users/ptien/tfds-download/apple2orange/experiment-0/generated_y/generated_from_"

    image_path = "/Users/ptien/tfds-download/horse2zebra/testA/"
    gan_image_path_1 = "/Users/ptien/tfds-download/horse2zebra/experiment-500000/generated_y/generated_from_"
    gan_image_path_2 = "/Users/ptien/tfds-download/horse2zebra/experiment-1000/generated_y/generated_from_"
    gan_image_path_3 = "/Users/ptien/tfds-download/horse2zebra/experiment-0/generated_y/generated_from_"
    
    image_paths = [image_path, gan_image_path_1, gan_image_path_2, gan_image_path_3]
    all_images = glob.glob(image_path + "*.jpg")
    batch_size = 10
    # file_name = ["n07740461_240.jpg", "n07740461_411.jpg", "n07740461_14960.jpg", 
    #              "n07740461_40.jpg", "n07740461_1690.jpg", "n07740461_12921.jpg"]

    # show_kd_plot(image_paths, file_name, filter_=False, output="gan_performance_a2o.jpg", 
    #          custom_std=False, iqr_choice=True)
    c = 0
    total_images = len(all_images)
    while len(all_images) > 0:
        print(c, total_images, len(all_images))
        batch_images = all_images[:batch_size]
        batch_images = [ f.split("/")[-1] for f in batch_images]
        show_kd_plot(image_paths, batch_images, filter_=False, output="gan_performance_h2z_batch_{c}.jpg".format(c=c),
                 custom_std=False, iqr_choice=True)
        all_images = all_images[batch_size:]
        c += 1
