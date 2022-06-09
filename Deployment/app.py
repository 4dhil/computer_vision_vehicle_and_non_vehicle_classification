import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('model1.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Vehicle and Non Vehicle Detection
         """
         )

file = st.file_uploader("Upload your Vehicle or Non Vehicle Image!!", type=["jpg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
        size = (64, 64)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    if predictions[0][1] == 1:
        st.write('vehicle')
    elif predictions[0][0] == 1:
        st.write('nonvehicle')