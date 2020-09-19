import shap
import numpy as np
import tensorflow as tf
import os
from glob import glob




def shap_numy(path, all_classes, model):
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
            # background_image_path = "/home/ptien/from_ss/mnist/0/transform_to_0/0_transformed.png"
            for j in all_classes:
                background_image_path = os.path.join(base_image_path, f"transform_to_{j}",
                                                     f"{i}_transformed.png")
                if c == 1:
                    imag_obj = tf.keras.preprocessing.image.load_img(input_image_path, target_size=(h, w), color_mode="grayscale")
                    background_obj = tf.keras.preprocessing.image.load_img(background_image_path, target_size=(h, w), color_mode="grayscale")
                else:
                    imag_obj = tf.keras.preprocessing.image.load_img(input_image_path, target_size=(h, w))
                    background_obj = tf.keras.preprocessing.image.load_img(background_image_path, target_size=(h, w))
                imag = np.array([tf.keras.preprocessing.image.img_to_array(imag_obj)/normalized_factor])
                background = np.array([tf.keras.preprocessing.image.img_to_array(background_obj)/normalized_factor])
                e = shap.DeepExplainer(model, background)
                import pdb; pdb.set_trace()
                shap_values = e.shap_values(imag)
                output_path = os.path.join(base_image_path, f"transform_to_{j}", "output", "explains", "numpy", "shap")
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                f_in_path = os.path.join(output_path, f"{i}_shap.npy")
                np.save(f_in_path, shap_values)


if __name__ == '__main__':
    model = tf.keras.models.load_model("/home/ptien/from_ss/mnist/model_mnist_keras.h5")
    # model = tf.keras.models.load_model("/home/ptien/from_ss/mnist1/model_mnist1_keras.h5")
    all_classes = [ i for i in range(10)]
    # all_classes = [1, 4, 7]
    # path = "/home/ptien/from_ss/mnist/"
    path = "/home/ptien/from_ss/shap2/mnist/"
    # path = "/home/ptien/from_ss/mnist1_n"
    # path = "/home/ptien/from_ss/mnist1_r"
    # path = "/home/ptien/from_ss/mnist1_s"
    # path = "/home/ptien/from_ss/mnist1_rs/"
    shap_numy(path, all_classes, model)


# input_image_path = "/home/ptien/from_ss/mnist/0/0.png"
# background_image_path = "/home/ptien/from_ss/mnist/0/transform_to_0/0_transformed.png"
# h, w , c= [28, 28, 1]
# normalized_factor = 255
# # imag = tf.keras.preprocessing.image.load_img(input_image_path, target_size=(h, w), color_mode="grayscale")
# imag_obj = tf.keras.preprocessing.image.load_img(input_image_path, target_size=(h, w), color_mode="grayscale")
# imag = np.array([tf.keras.preprocessing.image.img_to_array(imag_obj)/normalized_factor])
# background_obj = tf.keras.preprocessing.image.load_img(background_image_path, target_size=(h, w), color_mode="grayscale")
# background = np.array([tf.keras.preprocessing.image.img_to_array(background_obj)/normalized_factor])
# e = shap.DeepExplainer(model, background)
# shap_values = e.shap_values(imag)
# print("pass")
