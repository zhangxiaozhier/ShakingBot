import cv2
import numpy as np
import skimage
depth_filename="./data_studio_painted_bag/depth_1_0.png"
imgOri = cv2.imread(depth_filename, -1)
depth=cv2.split(imgOri)[0]
depth[depth>2000]=0
depth=depth/2000.0000
cv2.imshow('imgOri',depth)
cv2.imwrite("./data_studio_painted_bag/show_depth.png", depth.astype(np.uint16) )
cv2.waitKey(0)