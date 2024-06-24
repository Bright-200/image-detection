import tensorflow  as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
##generatino image from places...
model=load_model('Image_classification.keras')
data_cat=['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']
img_height=180
img_width=180

imaged =st.text_input('The Image name is ')

if (imaged==''):
    st.image('banana.jpg',width=200)
    st.write("Please this is a default image place inset one to predict your image for you")    
else:
    image=imaged
    image_load=tf.keras.utils.load_img(image,target_size=(img_height,img_width))
    img_arr=tf.keras.utils.array_to_img(image_load)
    img_bat=tf.expand_dims(img_arr,0)
    st.header("Image Classification Model")
    predict=model.predict(img_bat)

    score=tf.nn.softmax(predict)
    st.image(image,width=200)
    st.write('Veg/Fruit in image is '+ data_cat[np.argmax(score)])
    st.write('With accuracy of '+ str(np.max(score)*100))