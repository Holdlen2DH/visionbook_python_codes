"""
demo of imashapen Image sharping.

by Holdlen2DH 2017-11-29
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.lines as lines

plt.style.use('seaborn')

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

gy = np.gradient(y)

lap_y = np.gradient(gy)
lap_ysc = C * lap_y

y_sharp = y - lap_ysc
print(y_sharp)
y_sharp_bound = y_sharp.copy()
y_sharp_bound[y_sharp > 255] = 255
y_sharp_bound[y_sharp < 0] = 0

ymin = np.min(y_sharp)
ymax = np.max(y_sharp)

fig, [ax1, ax2, ax3, ax4]= plt.subplots(4, 1)

# subplot 1
ax1.plot(x, y, "-", linewidth = 2)
# ax1.axis([x[0], x[x.size - 1], ymin, ymax])
l1 = lines.Line2D([do_low, do_low], [ymin, ymax], color = 'k', linewidth = 1)
l2 = lines.Line2D([do_high, do_high], [ymin, ymax], color = 'k', linewidth = 1)
ax1.add_line(l1)
ax1.add_line(l2)
ax1.set_title("Original signal $\it f$")

# subplot 2
ax2.plot(x, gy, "-", linewidth = 2)
ax2.set_ylim([-50, 50])
l1 = lines.Line2D([do_low, do_low], 1.1 * np.array([np.min(gy), np.max(gy)]), color = 'k', linewidth = 1)
l2 = lines.Line2D([do_high, do_high], 1.1 * np.array([np.min(gy), np.max(gy)]), color = 'k', linewidth = 1)
ax2.add_line(l1)
ax2.add_line(l2)
ax2.set_title("First derivative $\it \partial f / \partial x$")

# subplot 3
ax3.plot(x, lap_ysc, "-", linewidth = 2)
ax3.set_ylim([-100, 100])
l1 = lines.Line2D([do_low, do_low], 1.1 * np.array([np.min(lap_ysc), np.max(lap_ysc)]), color = 'k', linewidth = 1)
l2 = lines.Line2D([do_high, do_high], 1.1 * np.array([np.min(lap_ysc), np.max(lap_ysc)]), color = 'k', linewidth = 1)
ax3.add_line(l1)
ax3.add_line(l2)
ax3.set_title("Second derivative-Laplacian(scaled) $\it C \partial^2 f / \partial^2 x$")

# subplot 4
ax4.plot(x, y, 'b-', linewidth = 1, label = "original signal")
ax4.plot(x, y_sharp, "g-", linewidth = 4, label = "sharpened signal")
ax4.plot(x, y_sharp_bound, "r-", linewidth = 2, label = "truncated to <0, 255>")
ax4.legend()
l1 = lines.Line2D([do_low, do_low], [ymin, ymax], color = 'k', linewidth = 1)
l2 = lines.Line2D([do_high, do_high], [ymin, ymax], color = 'k', linewidth = 1)
ax4.add_line(l1)
ax4.add_line(l2)
ax4.set_title("Improved signal $f - \it C \partial^2 f / \partial^2 x$")


plt.tight_layout()
plt.show()


# img = Image.open(current_dir + "/images/patterns.png")
# print(img.format, img.size, img.mode)

# img_data = np.asarray(img, dtype = np.uint8)

# fig = plt.figure()
# plt.imshow(img_data)
# plt.show()
