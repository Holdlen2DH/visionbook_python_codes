"""
Demo showing the useage of imthresh.

the python codes by Holdlen2DH 2017-12-13 holdlen2dh@126.com
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.lines as lines

current_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = current_dir + "/output_images"

def imthresh(img):
    """
    Iterative (optimal) image thresholding. Threshhold a grayscale image using 
    an automatic iterative threshold selection.
    """
    img_vecs = img.flatten()

    # pre-calculate the histogram and cumulative histogram.
    vbins = np.arange(0, 257, 1)
    img_hist, hist_edges = np.histogram(img_vecs, vbins)
    vbins = (hist_edges[:-1] + hist_edges[1:])/2
    
    hist_times_gray = np.cumsum(img_hist * np.arange(0, 256, 1))
    cum_hist = np.cumsum(img_hist)

    # A first approximation of the background mean mean_1 is the mean of the corner pixels.
    # The third corner's index seems to be wrong!
    m, n = img.shape
    sum_bg = np.sum(img_vecs[[0, n - 1, n * (m - 1), m * n - 1]])
    num_pix_bg = 4
    mean1 = sum_bg/4
    mean2 = (np.sum(img_vecs) - sum_bg)/(m *n - num_pix_bg)
    threshold_val = np.uint8(np.ceil((mean1 + mean2)/2))


    if (threshold_val != 0) and (cum_hist[threshold_val - 1] == 0):
        threshold_val_old = threshold_val

    threshold_val_old = 0 # weird
    while threshold_val != threshold_val_old:
        threshold_val_old = threshold_val
        mean1 = hist_times_gray[threshold_val - 1]/cum_hist[threshold_val - 1]
        mean2 = (hist_times_gray[-1] - hist_times_gray[threshold_val - 1])/(cum_hist[-1] - cum_hist[threshold_val - 1])

        threshold_val = np.uint8(np.ceil((mean1 + mean2)/2))


    img_out = img >= threshold_val
    return img_out, threshold_val

img = Image.open(current_dir + "/images/figures2.jpg")
print(img.format, img.size, img.mode)

img_data = np.asarray(img, dtype = np.uint8)

# threshold
img_out, threshold_val = imthresh(img_data)

plt.figure()
plt.imshow(img_data, cmap = "gray")
plt.title("Original image")
plt.savefig(output_dir + "/imthresh_input.jpg")


plt.figure()
plt.imshow(img_out, cmap = "gray")
plt.title("Threshold segmentation")
plt.savefig(output_dir + "/imthresh_output.jpg")


fig, ax = plt.subplots()
ax.hist(img_data.flatten(), np.arange(0, 257, 1), label = "Histogram")
vbins = np.arange(0, 257, 1)
img_hist, hist_edges = np.histogram(img_data.flatten(), vbins)
ymax = np.max(img_hist)
l1 = lines.Line2D([threshold_val, threshold_val], [0, ymax], color = 'r', linewidth = 3, label = "Threshold")
ax.add_line(l1)
ax.legend()


plt.title("Histogram and threshold value")
plt.savefig(output_dir + "/imthresh_histogram.jpg")
plt.show()


