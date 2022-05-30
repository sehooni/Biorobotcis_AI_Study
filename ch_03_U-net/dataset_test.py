import cv2
from PIL import Image
import numpy as np
import h5py


path_img = ""

for i in range(1,10000): # 테스트셋에 있는 사진만큼

   dim = (256, 256) #(w,h)
   image = cv2.imread(path_img + "resize-img-" + str(i) + ".png", 1)
   resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
   cv2.imwrite(path_img + "resize-img-" + str(i) + ".png", resized)

images = []

images = np.array(images)

with h5py.File("Dataset_train.h5", 'w') as hdf:
    hdf.create_dataset('images', data=images, compression='gzip', compression_opts=9)
