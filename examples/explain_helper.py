import pandas as pd
import PIL
import tensorflow as tf
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath('..'))
from deepexplain.tensorflow import DeepExplain


def load_image(image_path, base_image_path, file_name, labels, num_class):
    assert len(labels) == len(file_name), "length of labels and file_name should be the same."
    image_array = []
    base_image_array = []
    for f in file_name:
        # imag_path =  "/Users/ptien/tfds-download/apple2orange/testA/" + f
        # base_path = "/Users/ptien/tfds-download/apple2orange/experiment2-500000/generated_y/generated_from_" + f
        imag_path = image_path + f
        base_path = base_image_path + "generated_from_" + f
        imag_temp = tf.keras.preprocessing.image.load_img(imag_path, target_size=(256, 256, 3))
        base_imag_temp = tf.keras.preprocessing.image.load_img(base_path, target_size=(256, 256, 3))
        input_np= tf.keras.preprocessing.image.img_to_array(imag_temp)/255.
        base_input_np = tf.keras.preprocessing.image.img_to_array(base_imag_temp)/255.
        image_array += [input_np]
        base_image_array += [base_input_np]
    imag = np.array(image_array, dtype=np.float)
    base_imag = np.array(base_image_array, dtype=np.float)
    y_label = labels # apple
    return imag, y_label, base_imag

def explain_model(model_path, imag, y_label, num_class, base_imag, explain_types, steps=100):
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
                attributions_base_ = de.explain(type_.split("_base")[0], target_tensor, input_tensor, xs, ys=ys, baseline=base_imag, steps=steps)
                output[type_] = attributions_base_
            else:
                attributions_ = de.explain(type_, target_tensor, input_tensor, xs, ys=ys)
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


if __name__ == '__main__':
    image_path = "/Users/ptien/tfds-download/apple2orange/testA/"
    base_image_path = "/Users/ptien/tfds-download/apple2orange/experiment2-500000/generated_y/"
    file_name = ["n07740461_10011.jpg", "n07740461_240.jpg", "n07740461_14960.jpg", "n07740461_2770.jpg"]
    model_path = "/Users/ptien/DeepLearning/research/gan/apple2orange.h5"
    num_class = 2
    labels = [0,0,0,0]
    imag, y_label, base_imag = load_image(image_path, base_image_path, file_name, labels, num_class)
    # (model_path, imag, y_label, num_class, base_imag)
    attributions_ig, attributions_ig_base_line, attributions_dl, attributions_dl_base_line = explain_model(model_path, imag, y_label, num_class, base_imag)
    
