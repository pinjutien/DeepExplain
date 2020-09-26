import shap
import numpy as np
import tensorflow as tf
import os
from glob import glob




def shap_numy(path, all_classes, model, baseline_type):
    # path = "/home/ptien/from_ss/mnist/"
    h, w , c= [28, 28, 1]
    # h, w , c= [28, 28, 3]
    normalized_factor = 255
    for i in all_classes:
        all_imag = glob(os.path.join(path, f"{i}", "*.png"))
        print(f"label: {i}")
        for input_image_path in all_imag:
            # imag_path = os.path.join(path, f"{i}", f"{i}.png")
            base_image_path = os.path.dirname(input_image_path)
            file_name = os.path.basename(input_image_path).split(".")[0]
            # background_image_path = "/home/ptien/from_ss/mnist/0/transform_to_0/0_transformed.png"
            for j in all_classes:
                # background_image_path = os.path.join(base_image_path, f"transform_to_{j}",
                #                                      f"{i}_transformed.png")
                background_image_path = os.path.join(path, "baseline", baseline_type, 
                                                     f"{file_name}_transformed_{j}.png")
                if not os.path.exists(background_image_path):
                    continue
                if c == 1:
                    imag_obj = tf.keras.preprocessing.image.load_img(input_image_path, target_size=(h, w), color_mode="grayscale")
                    background_obj = tf.keras.preprocessing.image.load_img(background_image_path, target_size=(h, w), color_mode="grayscale")
                else:
                    imag_obj = tf.keras.preprocessing.image.load_img(input_image_path, target_size=(h, w))
                    background_obj = tf.keras.preprocessing.image.load_img(background_image_path, target_size=(h, w))
                imag = np.array([tf.keras.preprocessing.image.img_to_array(imag_obj)/normalized_factor])
                background = np.array([tf.keras.preprocessing.image.img_to_array(background_obj)/normalized_factor])
                e = shap.DeepExplainer(model, background)
                shap_values = e.shap_values(imag)

                output_path = os.path.join(base_image_path,  "output", "shap", baseline_type)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                f_in_path = os.path.join(output_path, f"{file_name}_baseline_{j}.npy")
                np.save(f_in_path, shap_values)
                image_path = os.path.join(output_path, f"{file_name}_baseline_{j}.png")
                shap.image_plot(shap_values, -imag, path=image_path)
                image_path_1 = os.path.join(output_path, f"{file_name}_baseline_{j}_v1.png")                
                shap.image_plot([shap_values[i], shap_values[j]], -imag, path=image_path_1)
                image_path_2 = os.path.join(output_path, f"{file_name}_baseline_{j}_v2.png")
                shap.image_plot([shap_values[i]- shap_values[j]], -imag, path=image_path_2)



def shap_image(target_label, baseline_label, base_dir, output_path):
    # shap_path = f"/Users/pin-jutien/tfds-download/from_ss/shap2/mnist/{target_label}/transform_to_{baseline_label}/output/explains/numpy/shap/{target_label}_shap.npy"

    shap_values = np.load(shap_path)[:, 0, :, :, :]
    input_image_path = f"/Users/pin-jutien/tfds-download/from_ss/shap2/mnist/{target_label}/{target_label}.png"
    background_image_path = f"/Users/pin-jutien/tfds-download/from_ss/shap2/mnist/{target_label}/transform_to_{baseline_label}/{target_label}_transformed.png"
    if c == 1:
        imag_obj = tf.keras.preprocessing.image.load_img(input_image_path, target_size=(h, w), color_mode="grayscale")                           
        background_obj = tf.keras.preprocessing.image.load_img(background_image_path, target_size=(h, w), color_mode="grayscale")                
    else:                                                                                                                                        
        imag_obj = tf.keras.preprocessing.image.load_img(input_image_path, target_size=(h, w))                                                   
        background_obj = tf.keras.preprocessing.image.load_img(background_image_path, target_size=(h, w))                                        
    imag = np.array([tf.keras.preprocessing.image.img_to_array(imag_obj)/normalized_factor])                                                     
    background = np.array([tf.keras.preprocessing.image.img_to_array(background_obj)/normalized_factor]) 
    num_class = shap_values.shape[0]
    shap_values = [shap_values[i].reshape(1, 28, 28, 1) for i in range(num_class)]
    # imag_path = f"/Users/pin-jutien/tfds-download/from_ss/shap/mnist/shap_value/result/"
    imag_path = os.path.join(output_path, "image",f"shap_target_{target_label}_baseline_{baseline_label}.png")
    np_path = os.path.join(output_path, "numpy",f"shap_target_{target_label}_baseline_{baseline_label}.npy")
    np.save(np_path, np.array(shap_values))
    shap.image_plot(shap_values, -imag, path=imag_path)

if __name__ == '__main__':
    model = tf.keras.models.load_model("/home/ptien/from_ss/mnist/model_mnist_keras.h5")
    # model = tf.keras.models.load_model("/home/ptien/from_ss/mnist1/model_mnist1_keras.h5")
    all_classes = [ i for i in range(10)]
    # all_classes = [1, 4, 7]
    # path = "/home/ptien/from_ss/mnist/"
    # path = "/home/ptien/from_ss/shap2/mnist/"
    path = "/home/ptien/from_ss/2020-09-20/mnist"
    baseline_type = "gan"
    # path = "/home/ptien/from_ss/mnist1_n"
    # path = "/home/ptien/from_ss/mnist1_r"
    # path = "/home/ptien/from_ss/mnist1_s"
    # path = "/home/ptien/from_ss/mnist1_rs/"
    shap_numy(path, all_classes, model, baseline_type)
    # output_path = os.path.join(path, "shap", "plot")
    # shap_image(target_label, baseline_label, output_path)
