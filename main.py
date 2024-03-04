import streamlit as st
import numpy as np
import time
import cv2
import os
from os import listdir
from os.path import isfile, join, abspath, dirname
from matplotlib import pyplot as plt 
from PIL import Image
import base64

# Define base directory based on script location
base_dir = os.path.dirname(abspath(__file__))

# Paths relative to the script's location
model_file_path = os.path.join(base_dir,'NeuralStyleTransfer', 'models')
output_folder = os.path.join(base_dir, 'output')
art_path = os.path.join(base_dir,'NeuralStyleTransfer', 'art')
style_names = ['candy','compose','feather','muse','mosaic','stary','scream','wave','udnie']

def genrate_images(uploaded_image_path):
    img = cv2.imread(uploaded_image_path)
    model_file_paths = [f for f in listdir(model_file_path) if isfile(join(model_file_path, f))]
    for (i,model) in enumerate(model_file_paths):
        print(str(i+1) + ". Using Model: " + str(model)[:-3]) 
        style = cv2.imread(join(art_path, str(model)[:-3] + ".jpg"))
        neuralStyleModel = cv2.dnn.readNetFromTorch(join(model_file_path, model))
        height, width = int(img.shape[0]), int(img.shape[1])
        newWidth = int((640 / height) * width)
        resizedImg = cv2.resize(img, (newWidth, 640), interpolation = cv2.INTER_AREA)
        inpBlob = cv2.dnn.blobFromImage(resizedImg, 1.0, (newWidth, 640), (103.939, 116.779, 123.68), swapRB=False, crop=False)
        neuralStyleModel.setInput(inpBlob)
        output = neuralStyleModel.forward()
        output = output.reshape(3, output.shape[2], output.shape[3])
        output[0] += 103.939
        output[1] += 116.779
        output[2] += 123.68
        output /= 255
        output = output.transpose(1, 2, 0)
        output = (output * 255).astype(np.uint8)
        output_filename = f"{model[:-3]}{i+1}.png"
        cv2.imwrite(join(output_folder, output_filename), output)

# Streamlit interface
st.title(':orange_book: NEURAL STYLE :sunglasses:')
st.header('Upload an image and apply textures over it.')

# Uploading files
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
uploaded_image_path = "input_images\uploaded_image.jpg"

# Generate images
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True, width=10)
    img.save(uploaded_image_path)
    if st.button("Generate"):
        with st.spinner('Generating...'):
            genrate_images(uploaded_image_path)
        st.success("Image generated successfully.")

# Display and download generated images
if uploaded_file is not None and st.button("Download files"):
    output_files = os.listdir(output_folder)
    num_files = len(output_files)
    col = st.columns(min(num_files, 9))
    for i, file in enumerate(output_files[:len(col)]):
        file_path = os.path.join(output_folder, file)
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                contents = f.read()
                data_url = base64.b64encode(contents).decode("utf-8")
                href = f'<a href="data:file/jpg;base64,{data_url}" download="{file}">{style_names[i]}</a>'
                show_img = Image.open(file_path)
                col[i].image(show_img, use_column_width=True)
                col[i].markdown(href, unsafe_allow_html=True)
