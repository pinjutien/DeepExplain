from skimage import feature, transform
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from scipy.stats import iqr


def plot(data, xi=None, cmap='RdBu_r', axis=plt, percentile=100, dilation=3.0, alpha=0.8):
    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, data.shape[1], dx)
    yy = np.arange(0.0, data.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)
    overlay = None
    if xi is not None:
        # Compute edges (to overlay to heatmaps later)
        xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
        in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(data), percentile)
    abs_min = abs_max

    if len(data.shape) == 3:
        data = np.mean(data, 2)
    axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    if overlay is not None:
        axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
    axis.axis('off')
    return axis



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
