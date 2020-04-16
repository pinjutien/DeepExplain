from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
import PIL
from deepexplain.tensorflow import DeepExplain
import tensorflow as tf
import numpy as np
from explain_helper import explain_model, load_image
import glob
from utils import plot, plt

image_path = "/Users/ptien/tfds-download/apple2orange/testA/"
base_image_path = "/Users/ptien/tfds-download/apple2orange/experiment2-500000/generated_y/"
# file_name = ["n07740461_10011.jpg", "n07740461_240.jpg", "n07740461_14960.jpg", 
#              "n07740461_2770.jpg", "n07740461_14600.jpg", "n07740461_14531.jpg", "n07740461_14300.jpg"]
model_path = "/Users/ptien/DeepLearning/research/gan/apple2orange.h5"
num_class = 2
# labels = [0]*len(file_name) # [0,0,0,0,0,0]
explain_types = ["intgrad", "intgrad_base", "deeplift", "deeplift_base"]
# stochastic_mask_flag = False
# imag, y_label, base_imag = load_image(image_path, base_image_path, file_name, labels, num_class)
# output = explain_model(model_path, imag, y_label, num_class, base_imag, explain_types,
#                        steps=150,
#                        stochastic_mask_flag=stochastic_mask_flag)



def explains_methods(image_path, base_image_path, file_name, num_class, explain_types,labels=None, stochastic_mask_flag=False, steps=150, path=None):
    if not labels: labels = [0]*len(file_name)
    # explain_types = ["intgrad", "intgrad_base", "deeplift", "deeplift_base"]
    imag, y_label, base_imag = load_image(image_path, base_image_path, file_name, labels, num_class)
    output = explain_model(model_path, imag, y_label, num_class, base_imag, explain_types,
                           steps=steps,
                           stochastic_mask_flag=stochastic_mask_flag)
    if path:
        for method in output.keys():
            for i in range(len(file_name)):
                imag_name = file_name[i]
                arr = output[method][i]
                np.save(path+ "{method}_{i}_{imag_name}.npy".format(method=method, i=i, imag_name=imag_name), arr)
    return output, imag, base_imag


def plot_explains(output, imag, base_imag, file_name, explain_types, bc, path=None):
    nrows = len(file_name)
    ncols = 2 + len(explain_types) # 6
    xs = imag
    attributions_ig = output["intgrad"]
    attributions_ig_base_line = output["intgrad_base"]
    attributions_dl = output["deeplift"]
    attributions_dl_base_line = output["deeplift_base"]
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*ncols, 3*nrows))
    axes = axes.reshape(nrows, ncols)
    plot_info = []
    for i in range(nrows):
        plot_info += [[i, file_name[i]]]
        a1 = attributions_ig[i]
        a2 = attributions_ig_base_line[i]
        a3 = attributions_dl[i]
        a4 = attributions_dl_base_line[i]
        plot(xs[i], axis=axes[i,0]).set_title('Original')
        plot(base_imag[i], axis=axes[i,1]).set_title('transformed')
        plot(a1, axis=axes[i,2]).set_title('intgrad')
        plot(a2, axis=axes[i,3]).set_title('intgrad_base')
        plot(a3, axis=axes[i,4]).set_title('deeplift')
        plot(a4, axis=axes[i, 5]).set_title('deeplift_base')
    if path:
        os.makedirs(path + "plots/", exist_ok=True)
        pd.DataFrame(plot_info, columns=["plot_id", "image_name"]).to_csv(path + "plots/plot_info_{x}.csv".format(x=bc))
        fig.savefig(path + "plots/explains_methods_{x}.jpg".format(x=bc))
        
image_pattern = "*.jpg"
all_images = glob.glob(image_path + image_pattern)[:11]
batch_size = 5
print("Number of image: {x}".format(x=len(all_images)))
stats = []
bc = 0
final_arr = np.array([])
path = "./experiment/"
os.makedirs(path, exist_ok=True)
while all_images:
    print("Remaining images: {x}".format(x=len(all_images)), end='\r', flush=True)
    batch_of_images = all_images[:batch_size]
    file_name = [b.split("/")[-1] for b in batch_of_images]
    arr_path = path + "batch/{bc}/".format(bc=bc)
    os.makedirs(arr_path, exist_ok=True)
    output, imag, base_imag = explains_methods(image_path, base_image_path, file_name, num_class, explain_types,labels=None, stochastic_mask_flag=False, steps=150, path=arr_path)
    plot_explains(output, imag, base_imag, file_name, explain_types, bc, path)
    batch_id = np.array([bc]*len(file_name)).reshape(-1,1)
    temp = np.hstack((np.array(file_name).reshape(-1, 1), batch_id))
    if len(final_arr) == 0:
        final_arr = temp
    else:
        final_arr = np.vstack((final_arr, temp))
    bc += 1
    all_images= all_images[batch_size:]
df = pd.DataFrame(final_arr, columns=["image", "batch_id"])
df.to_csv(path + "stats.csv")
print("end")
