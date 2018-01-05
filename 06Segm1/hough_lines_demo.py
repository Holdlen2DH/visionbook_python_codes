"""
Demo showing the useage of hough_lines.

the python codes by Holdlen2DH 2017-12-13 holdlen2dh@126.com
"""
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.lines as lines

import cv2

current_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = current_dir + "/output_images"

# read image via opencv
img = cv2.imread(current_dir + "/images/chess.jpg")
img = cv2.GaussianBlur(img,(3,3),0)  
img_cv_edge = cv2.Canny(img, 0.02, 2.5)
img_edge = cv2.bitwise_and(img, img, img_cv_edge)


plt.figure()
plt.imshow(img, cmap = "gray")
plt.title("Original image")
plt.savefig(output_dir + "/hough_input.jpg")

plt.figure()
cv2.imshow('Canny', img_edge)
cv2.waitKey(0)  
plt.imshow(img_edge)
plt.title("Canny edges")
plt.savefig(output_dir + "/hough_edges.jpg")