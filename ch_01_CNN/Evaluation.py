from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageFile
import os


model_1 = load_model('Saved_model/ResNET50_Flow.h5')

img = image.load_img('test_files/111.jpg', target_size=(244, 244))
plt.imshow(img)
x=image.img_to_array(img)
x=np.expand_dims(x, axis=0)
x=preprocess_input(x)
classes = model_1.predict(x)
print(classes)
result = list(classes[0]).index(max(classes[0]))
print(result)

if result == 0:
    print('this flower is daisy')
elif result == 1:
    print('this flower is dandelion')
elif result == 2:
    print('this flower is rose')
elif result == 3:
    print('this flower is sunflower')
elif result == 4:
    print('this flower is tulip')
plt.show()