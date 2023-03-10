import os   
import cv2 
import numpy as np
import os
import time

rgb_path ='./colorizedDepth/' 
depth_path = './depth/'
out_rgb_path = './final_rgb/'
out_depth_path = './final_npy/'
rgb_count = 0
depth_count = 0
output_index = 0
abandon_index = 0

for rgb_file in os.listdir(rgb_path):
    rgb_count = rgb_count + 1
for depth_file in os.listdir(depth_path):
    depth_count = depth_count + 1
if rgb_count == depth_count:
    print("files count:",rgb_count)
    print("start...\n")
    time.sleep(1)
    count = rgb_count
    for i in range(0,count):
        print('Processing: ',i)
        rgb_name = rgb_path + 'CD_' + str(i) + '.jpg'
        depth_name = depth_path + 'depth_'  + str(i) + '.png'
        rgb_data = cv2.imread(rgb_name)
        depth_data = cv2.imread(depth_name)
        depth_info=cv2.split(depth_data)[0]
        a=int(83) # y start
        b=int(338) # y end
        c=int(133)
        d=int(376) # x end
        final_rgb = rgb_data[a:b,c:d]   
        final_depth = depth_info[a:b,c:d]   
        final_depth = final_depth /1000
        if np.sum(final_depth==0) > 10:
            abandon_index = abandon_index + 1
            continue
        cv2.imwrite(out_rgb_path + 'rgb_' + str(output_index) +'.png',final_rgb)
        np.save(out_depth_path  + str(output_index) + '_depth' + '.npy',final_depth)
        output_index = output_index + 1
        print("save :",output_index)
    
 
print('total number:',rgb_count)
print("abandon number:",abandon_index)   
print("finish number:",rgb_count - abandon_index)