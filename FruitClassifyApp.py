import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

model=tf.keras.models.load_model(
    "best_resnet.keras",
    compile=False,
    custom_objects={"ResNet50": ResNet50})

class_labels=['Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado', 'Banana', 'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Cherry', 'Clementine', 'Corn', 'Cucumber Ripe',
    'Grape Blue', 'Kiwi', 'Lemon', 'Limes', 'Mango', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Pear', 'Pepper Green', 'Pepper Red', 'Pineapple', 'Plum',
    'Pomegranate', 'Potato Red', 'Raspberry','Strawberry', 'Tomato', 'Watermelon']

st.title("Capture an image")
img=st.camera_input("Take a picture!!")

if img is not None:
    pil_img=Image.open(img).convert("RGB")
    img_resize=pil_img.resize((224, 224))
    img_arr=image.img_to_array(img_resize)
    img_arr=np.expand_dims(img_arr, axis=0)
    img_arr=preprocess_input(img_arr)
    with st.spinner("Processing..."):
        preds=model.predict(img_arr)
        pred_index=np.argmax(preds, axis=1)[0]
        pred_label=class_labels[pred_index]
    st.success(f"Prediction: {pred_label}")
    fig, ax=plt.subplots()
    ax.imshow(pil_img)
    ax.axis("off")
    st.pyplot(fig)
