from keras.models import load_model
model = load_model(r"C:\Users\SMD TEJA\Downloads\DenseNet121-eye_disease-96.20.h5")
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from tensorflow.keras.models import Model
from keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

def test(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    pred = model.predict(x)
    predict = np.argmax(pred, axis=1)
    index = [ 'cataract', 'diabetic_retinopathy','glaucoma', 'normal']
    result = str(index[predict[0]])
    print(result)



img =image.load_img(r"C:\python\academic project\data\eye_disease\glaucoma\_0_4517448.jpg",target_size=(224,224))

test(img)
