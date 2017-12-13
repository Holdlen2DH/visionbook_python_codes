"""
Demo showing the useage of dpboundary.
As the matlab codes, it demonstrates dpboundary on the task of tracing
a blood vessel on part of an MRI image of a lower limb. Since the vessel 
is bright, it takes a negative value of the brightness as the cost function.

the python codes by Holdlen2DH 2017-12-13 holdlen2dh@126.com
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.lines as lines

current_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = current_dir + "/output_images"

def dpboundary():
    """
    Boundary tracing using dynamic programming.
    """


img = Image.open(current_dir + "/images/limb_vessels2.jpg")
print(img.format, img.size, img.mode)

img_data = np.asarray(img, dtype = np.uint8)

fig = plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img_data, cmap = "gray")
plt.title("Original image")
plt.savefig(output_dir + "/dpboundary_output.jpg")



