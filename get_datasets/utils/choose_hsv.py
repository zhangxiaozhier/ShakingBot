
import numpy as np
import cv2

def val(x):
    global Lower, Upper
    #获取滑动条的数值
    l_h = cv2.getTrackbarPos("LowH", "control")
    h_h = cv2.getTrackbarPos("highH", "control")
    l_s = cv2.getTrackbarPos("LowS", "control")
    h_s = cv2.getTrackbarPos("highS", "control")
    l_v = cv2.getTrackbarPos("LowV", "control")
    h_v = cv2.getTrackbarPos("highV", "control")
    Lower = np.array([l_h, l_s, l_v])  # 要识别颜色的下限
    Upper = np.array([h_h, h_s, h_v])  # 要识别的颜色的上限



def color_(img):
    imgHSV = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)
    #创建滑动条
    cv2.namedWindow("control")
    cv2.resizeWindow("control",640,240)
    cv2.createTrackbar("LowH", "control", 0, 179, val)
    cv2.createTrackbar("highH", "control", 179, 179, val)
    cv2.createTrackbar("LowS", "control", 0, 255, val)
    cv2.createTrackbar("highS", "control", 255, 255, val)
    cv2.createTrackbar("LowV", "control", 0, 255, val)
    cv2.createTrackbar("highV", "control", 255, 255, val)


    while True:
        mask = cv2.inRange(imgHSV, Lower, Upper)
        # mask是把HSV图片中在颜色范围内的区域变成白色，其他区域变成黑色
        white_mask = np.average(mask)
        white_mask = white_mask / 255 * 100
        # white_mask是该色域在区域中的占比值
        img_mask = cv2.bitwise_and(img,img,mask=mask)
        # img_mask获得原图中的指定颜色图像
        all_img_mask = np.vstack([white_array, img, img_mask])
        # all_img_mask将所有的图像拼接起来
        cv2.putText(all_img_mask, str(white_mask)[0:6], (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("test", all_img_mask)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    Lower = np.array([0, 0, 0])  # 要识别颜色的下限
    Upper = np.array([179, 255, 255])  # 要识别的颜色的上限
    white_array = np.zeros((50,200,3),np.uint8)
    white_array.fill(255)
    path = "./final_rgb/rgb_1_0.png"
    img = cv2.imread(path)
    img = cv2.resize(img,(200,200))
    color_(img)
