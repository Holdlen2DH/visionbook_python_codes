"""
demo of imashapen Image sharping.

by Holdlen2DH 2017-11-29
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.lines as lines

current_dir = os.path.dirname(os.path.realpath(__file__))

step = 1
x1 = np.arange(1, 101, step)
x2 = np.arange(x1[x1.size - 1] + step, 2 * x1[x1.size - 1] + step, step)
n = 10

do_low = x1[np.int(np.round(x1.size/2)) - 1]
do_high = x2[np.int(np.round(x1.size/2)) - 1]

C = 20
ymin = 0
ymax = 255
x1_low = (x1/do_low)**(2 * n)
y1 = 255 * x1_low/(1 + x1_low)
y2 = ymax - y1

x = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))

fig, [ax1, ax2, ax3, ax4]= plt.subplots(4, 1)

ax1.plot(x, y, "-", linewidth = 2)
l1 = lines.Line2D([do_low, do_low], [ymin, ymax], color = 'k', linewidth = 1)
l2 = lines.Line2D([do_high, do_high], [ymin, ymax], color = 'k', linewidth = 1)
ax1.add_line(l1)
ax1.add_line(l2)
# ax1.axis([x[0], x[x.size - 1], ymin, ymax])

plt.show()

# img = Image.open(current_dir + "/images/patterns.png")
# print(img.format, img.size, img.mode)

# img_data = np.asarray(img, dtype = np.uint8)

# fig = plt.figure()
# plt.imshow(img_data)
# plt.show()
