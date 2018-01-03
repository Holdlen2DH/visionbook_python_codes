"""
Demo showing the useage of dpboundary.
As the matlab codes, it demonstrates dpboundary on the task of tracing
a blood vessel on part of an MRI image of a lower limb. Since the vessel 
is bright, it takes a negative value of the brightness as the cost function.

dp means dynamic programming, which based on the principle of optimality.

the python codes by Holdlen2DH 2017-12-13 holdlen2dh@126.com
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.lines as lines

current_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = current_dir + "/output_images"

def dpboundary(IM):
    """
    Boundary tracing using dynamic programming.

    IM is a m-by-n scalar input image with values representing the cost
    of a path going through a pixel.
    the returned x is the x coordinates of the optimal path.
    """

    # initial the matrix c which will contain for each pixel the total cost
    # of an optimal path from the first row (y = 0). The matrix p corresponds
    # to the 'pointers'; It contain values 1, 2, or 3 meaning that the optimal
    # path reaches the current pixel (y, x) from pixel(y - 1, x), (y - 1, x + 1), 
    # or (y - 1, x - 1), respectively.
    M, N = IM.shape
    c = np.zeros((M, N))
    p = np.zeors((M, N), type = 'uint8')        # save memory by using 8bit integers
    c[0, :] = IM[0, :]

    # the first pass of the algorithm goes through the image matrix c from the first
    # to the last row. For each row of c, we assemble a matrix d; each row of d corresponds
    # to one alternative (no shift, left shift, right shift) and contains the cost of reaching
    # the current row of c.
    for i in range(1, M + 1, 1):
        c0 = c[i - 1, :]
       
        e1 = c0
        e2 = np.hstack((c0[1:], c0[-1]))
        e3 = np.hstack((c0[0], c0[0:-1]))

        e = np.vstack((e1, e2, e3))
        d = np.tile(c0, (3, 1)) + e 

    return x

c = np.arange(15).reshape(3, 5)
c0 = c[1, :]
print(c0)
e1 = c0
e2 = np.hstack((c0[1:], c0[-2]))
print(e2)
print(c0[0:-2])

img = Image.open(current_dir + "/images/limb_vessels2.jpg")
print(img.format, img.size, img.mode)

img_data = np.asarray(img, dtype = np.uint8)

fig = plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img_data, cmap = "gray")
plt.title("Original image")
plt.savefig(output_dir + "/dpboundary_output.jpg")





