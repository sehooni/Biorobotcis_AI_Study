import cv2
from PIL import Image
import numpy as np
import h5py


path_img = ""
path_mask = ""

for i in range(1,10000): # 데이터셋에 있는 사진 개수만큼

   dim = (256, 256) #(w,h)
   image = cv2.imread(path_img + "resize-img-" + str(i) + ".png", 1)
   resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
   cv2.imwrite(path_img + "resize-img-" + str(i) + ".png", resized)
    
   mask = cv2.imread(path_mask + "resize-img-" + str(i) + ".png", 0)
   resized_mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)
   mask_binary = cv2.threshold(resized_mask, 128, 255, cv2.THRESH_BINARY)[1]
   cv2.imwrite(path_mask + 'resize-img-' + str(i) + ".png", mask_binary)

images = []
masks = []

for i in range(1, 10000): # 데이터셋에 있는 사진 개수만큼

    img = Image.open(path_img + 'resize-img-' + str(i) + ".png")
    arr = np.array(img)
    images.append(arr)
    img = Image.open(path_mask + 'resize-img-' + str(i) + ".png")
    arr = np.array(img)
    arr = np.expand_dims(arr, -1)
    masks.append(arr)

images = np.array(images)
print(images.shape)
masks = np.array(masks)
print(masks.shape)

with h5py.File("Dataset_train.h5", 'w') as hdf:
    hdf.create_dataset('images', data=images, compression='gzip', compression_opts=9)
    hdf.create_dataset('masks', data=masks, compression='gzip', compression_opts=9)