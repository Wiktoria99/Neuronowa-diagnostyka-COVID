import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import tensorflow as tf

from tensorflow.keras.preprocessing import image_dataset_from_directory

IMAGE_DIRECTORY = "TU_PODAJ_SCIEZKE_DO_PLIKU"
MODEL_DIRECTORY = "./model.h5"
model = tf.keras.models.load_model(MODEL_DIRECTORY)

class_names = ["negative", "positive"]
IMG_SIZE = (160, 160)

def to_grayscale_3(image):
    if(len(image.shape)==2):
        image = tf.convert_to_tensor(np.expand_dims(image, axis=-1))
    else:
        image = tf.image.rgb_to_grayscale(image)
    image = tf.image.grayscale_to_rgb(image)
    return image
    
def predict(image):   
    image = image.resize(IMG_SIZE)
    image = np.array(image)
    image = to_grayscale_3(image)
    image = np.expand_dims(image, axis=0)
    
    prediction = model.predict(image).flatten()
    prediction = tf.nn.sigmoid(prediction)
    prediction = tf.where(prediction < 0.5, 0, 1)
    
    return class_names[prediction.numpy()[0]]

def load_image_from_file(file_dir):
    # Read image 
    img = Image.open(file_dir) 
    return img

def predict_image(img):
    prediction = predict(img)
    print("Result:")
    print(prediction)

def predict_and_plot(image_dir: str):
    img = load_image_from_file(image_dir)
    predict_image(img)
    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    predict_and_plot(IMAGE_DIRECTORY)