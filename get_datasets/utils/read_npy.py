import numpy as np
import cv2

# arr = np.array([[1, 2, 3],
#                [4, 5, 6]])
# np.save('weight.npy', arr)

loadData = np.load('./final_npy/15_depth.npy')
cv2.imshow('depth',loadData)
print("----type----")
print(type(loadData))
print("----shape----")
print(loadData.shape)
print("----data----")
print(loadData.max())
print(loadData.min())
print(loadData)
print(np.sum(loadData==0))

if 0 in loadData:
    print('!!!!')
    
cv2.waitKey(0)
