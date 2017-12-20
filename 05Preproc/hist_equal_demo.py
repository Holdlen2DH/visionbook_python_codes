"""
Demo of HIST_EQUAL, histogram equalization.

by Holdlen2DH 2017-12-20 holdlen2dh@126.com
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.lines as lines

current_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = current_dir + "/output_images"

def hist_equal(IM):
    """
    histogram equalization.
    input: IM is a grayscale image with a size of M * N.
    output parameters:
    IM_out is the equalized iamge.
    H: a vector with 256 elements. H is the histogram of the input image.
    Hc: a vector with 256 elements. Hc is the cumulative histogram of the input functions.
    T:  a vector with 256 elements. T is the transfromation of the intensity.
    """
    if IM.ndim > 1:
        print("Colour image in the input, HIST_EQUAL process it as grayscale.")
    M, N = IM.shape

    if IM.dtype.name == "uint8":
        levels = 2**8
    elif IM.dtype.name == "uint16":
        levels = 2**16
    elif IM.dtype.name == "float":
        levels = 2**8
        if np.max(IM.flatten()) > 1:
            IM = np.round(IM)
            IM[IM < 0] = 0
            IM[IM > 255] = levels - 1
        else:
            IM = np.uint8(np.round(IM * 255))

    # compute the histogram of the input image first
    vbins = np.arange(0, 257, 1)
    H, H_edges = np.histogram(IM.flatten(), vbins)
    # form the comulative image histogram Hc
    Hc = np.cumsum(H)

    # create the look-up table. 
    # normlizing the cumulative histogram to have inter values between [0, levels - 1]
    T = np.round( Hc * (levels - 1)/(M * N))

    IM_out = T[IM]

    return IM_out, H, Hc, T


img = Image.open(current_dir + "/images/raising_moon_gray_small.jpg")
img_data = np.asarray(img, dtype = np.uint8)

img_out, H, Hc, T = hist_equal(img_data)

# input image
plt.figure()
plt.imshow(img_data, cmap = "gray")
plt.title("Input image")
plt.savefig(output_dir + "/histeq_input.jpg")
# equalized image
plt.figure()
plt.imshow(img_out, cmap = "gray")
plt.title("Equalized image")
plt.savefig(output_dir + "/histeq_output.jpg")
# histogram of input image
plt.figure()
plt.bar(np.arange(0, 256, 1), H)
plt.title("Histogram of the input image")
plt.xlabel("intensity")
plt.ylabel("frequency")
plt.axis([0, 255, 0, np.max(H)])
plt.savefig(output_dir + "/histeq_histinput.jpg")
# cumulative histogram of the input image
plt.figure()
plt.plot(np.arange(0, 256, 1), Hc)
plt.grid()
plt.title("Cumulative histogram of the input image")
plt.xlabel("intensity")
plt.ylabel("cumulative sum of the occurence")
# intensity transformation
fig, ax = plt.subplots()
val = 45
ax.plot(np.arange(0, 256, 1), T, lw = 2)
ax.set_title("Intensity transformation (normalized comulative histogram)")
ax.set_xlabel("intensity in the input image")
ax.set_ylabel("intensity in the output image")
ax.stem([val - 1], [T[val - 1]], 'r-')
L1 = lines.Line2D([0, val - 1], [T[val - 1], T[val - 1]], color = 'r', linewidth = 1)
ax.add_line(L1)
ax.text(80, 150, "All pixels with intensity " + str(val - 1) + "will have", fontsize = 12)
ax.text(80, 125, "the intensity " + str(val - 1) + " in the output image", fontsize = 12)
ax.grid()
ax.axis([0, 255, 0, 260])
plt.savefig(output_dir + "/histeq_lookup.jpg")


plt.figure()
sc = 255/np.max(H)
plt.bar(np.arange(0, 256, 1), H * sc, label = "input histogram")
vbins = np.arange(0, 257, 1)
Ho, Ho_edges = np.histogram(img_out.flatten(), vbins)
sc = 255/np.max(Ho)
plt.barh(np.arange(0, 256, 1), Ho * sc, label = "output histogram")
plt.step(np.arange(0, 256, 1), T, 'r-', linewidth = 3, label = "transformation function")
plt.grid()
plt.legend()
plt.axis([0, 255, 0, 260])
plt.title("Chnage of the histogram")
plt.xlabel("intensity in the input image")
plt.ylabel("intensity in the output image")
plt.savefig(output_dir + "/histeq_histtransf.jpg")

# Histogram of the output image
plt.figure()
plt.bar(np.arange(0, 256, 1), Ho)
plt.title("Histogram of the output image")
plt.xlabel("intensity")
plt.ylabel("frequency")
plt.axis([0, 255, 0, np.max(H)])
plt.savefig(output_dir + "/histeq_outhist.jpg")
# both
plt.figure()
plt.bar(np.arange(0, 256, 1), H, label = "input")
plt.bar(np.arange(0, 256, 1), Ho, label = "output")
plt.title("Histogram of the input and output image")
plt.xlabel("intensity")
plt.ylabel("frequency")
plt.legend()
plt.axis([0, 255, 0, np.max(H)])
plt.savefig(output_dir + "/histeq_bothhists.jpg")

