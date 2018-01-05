"""
Demo showing the useage of dpboundary.
As the matlab codes, it demonstrates dpboundary on the task of tracing a blood 
vessel on part of an MRI image of a lower limb. Since the vessel is bright, it 
takes a negative value of the brightness as the cost function.

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

    IM is a m-by-n scalar input image with values representing the cost of a 
    path going through a pixel. 
    The returned x is the x coordinates of the optimal path.
    """

    # initial the matrix c which will contain for each pixel the total cost
    # of an optimal path from the first row (y = 0). The matrix p corresponds
    # to the 'pointers'; It contain values 1, 2, or 3 meaning that the optimal
    # path reaches the current pixel (y, x) from pixel(y - 1, x), (y - 1, x + 1), 
    # or (y - 1, x - 1), respectively.
    M, N = IM.shape
    c = np.zeros((M, N))
    p = np.zeros((M, N), dtype = 'uint8')   # save memory by using 8bit integers
    c[0, :] = IM[0, :]

    # the first pass of the algorithm goes through the image matrix c from the 
    # first to the last row. For each row of c, we assemble a matrix d; each row 
    # of d corresponds to one alternative (no shift, left shift, right shift) 
    # and contains the cost of reaching the current row of c. Note how the 
    # optimal choice is found in parallel for thw whole row using the vectorized 
    # min function.
    for i in range(1, M, 1):
        c0 = c[i - 1, :]
       
        
        e1 = c0
        e2 = np.hstack((c0[1:], c0[-1]))
        e3 = np.hstack((c0[0], c0[0:-1]))

        e = np.vstack((e1, e2, e3))
        d = np.tile(IM[i - 1, :], (3, 1)) + e 

        c[i, :] = np.min(d, axis = 0)
        p[i, :] = np.argmin(d, axis = 0)

    # The second part of the algorithm follows the optimal path from 'cheapest' 
    # node xpos in the last row back to the first row, using the information 
    # from p and creating x on the way. We take care not to leave the image 
    # boundaries.

    x = np.zeros((M, 1))
    # cost = np.min(c[M - 1, :])
    xpos = np.argmin(c[M  - 1, :])

    for i in range(M - 1, 0, -1):
        x[i, 0] = xpos
        if (p[i, xpos] == 1 and xpos < N - 1):
            xpos = xpos + 1
        elif p[i, xpos] == 2 and xpos > 0:
            xpos = xpos - 1
    
    x[0, 0] = xpos
    return x
img = Image.open(current_dir + "/images/limb_vessels2.jpg")
print(img.format, img.size, img.mode)

img_data = np.asarray(img, dtype = np.uint8)
fig = plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img_data, cmap = "gray")
plt.title("Original image")

x = dpboundary(-img_data.astype(np.float))

plt.subplot(1, 2, 2)
plt.imshow(img_data, cmap = "gray")
plt.plot(x.flatten(), np.arange(0, img_data.shape[0], 1), 'r-')
plt.title("tracing results")
plt.suptitle("blood vessel tracing via dynamic programming")
plt.savefig(output_dir + "/dpboundary_results.jpg")
plt.show()





