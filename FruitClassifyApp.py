import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from keras.models import load_model
from tensorflow.keras.preprocessing import image

model= load_model("model_alexnet.h5")
class_labels=['Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado', 'Banana', 'Blueberry',
               'Cactus fruit', 'Cantaloupe', 'Cherry', 'Clementine', 'Corn', 'Cucumber Ripe', 'Grape Blue',
                 'Kiwi', 'Lemon', 'Limes', 'Mango', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach',
                   'Pear', 'Pepper Green', 'Pepper Red', 'Pineapple', 'Plum', 'Pomegranate', 'Potato Red',
                     'Raspberry', 'Strawberry', 'Tomato', 'Watermelon']

st.title("Capture an image!!")
img=st.camera_input("Take a picture!!")

if img is not None:
    pil_img=Image.open(img).convert("RGB")
    img_resize=pil_img.resize((224,224))
    img_arr=image.img_to_array(img_resize)
    img_arr=np.expand_dims(img_arr, axis=0)
    img_array=np.expand_dims(img_arr,axis=0)
    with st.spinner("Processing..."):
        preds=model.predict(img_arr)
        pred_index=np.argmax(preds)
        pred_label=class_labels[pred_index]

    st.success(print(pred_label))
    plt.imshow(pil_img)
    plt.axis("off")
    st.pyplot(plt)
