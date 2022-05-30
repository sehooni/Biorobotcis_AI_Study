import cv2
import numpy as np

path_img = '' # U-net 거쳐 나온 사진이 있는 경로설정

for i in range(1,5): # 사진개수만큼 for문 반복횟수 수정

    mask = cv2.imread(path_img + str(i) + "_pred.png", 0)
    mask_binary = cv2.threshold(mask, 40, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(path_img + 'post_' + str(i) + ".png", mask_binary)