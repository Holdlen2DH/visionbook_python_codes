"""
demo of imashapen Image sharping.

by Holdlen2DH 2017-11-29
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))

step = 1
x1 = np.arange(1, 101, 1)

img = Image.open(current_dir + "/images/patterns.png")
print(img.format, img.size, img.mode)

img_data = np.asarray(img, dtype = np.uint8)

plt.imshow(img_data)
plt.show()
