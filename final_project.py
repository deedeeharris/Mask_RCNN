
import streamlit as st
import cv2
import numpy as np
import skimage.io as io
import os
# import streamlit_extras
# from streamlit_extras.colored_header import colored_header
import glob



def image_resize(image, width=None, height=None, inter = cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = width/float(w)
        dim = (int(w*r),height)
    
    else:
        r = width/float(w)
        dim = (width, int(h*r))
        
    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)
    
    return resized


# def perc_cal(sum1,ripes):
#     if sum==0:
#         percent=0
#     else:
#     percent=sum1/ripes
#     percent=percent/100
#    return percent





page_icon = "web/favicon.png"



st.set_page_config(page_title='bell pepper helper', page_icon = page_icon, layout = 'wide', initial_sidebar_state = 'auto')
st.title('Bell pepper helper')



# side bar
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
        width: 350px
    }
    
    [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
        width: 350px
        margin-left: -350px
    }    
    </style>
    
    """,
    unsafe_allow_html=True,


)




st.sidebar.title('Let\'s Start!')


# add dropdown to select pages on left
app_mode = st.sidebar.selectbox('',
                                  ['About App', 'Analyse a plant'])








# About page
if app_mode == 'About App':
    
    
    
    # side bar
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
            width: 350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
            width: 350px
            margin-left: -350px
        }    
        </style>

        """,
        unsafe_allow_html=True,


    )

    st.markdown('''
                ## About our project \n
                #### Hey, we are Ran and Orel and we decided to develope a tool for bell pepper's farmers. \n
                #### Our concept is to help red bell pepper's farmer to decide when picking the fruits. \n
                #### By uploading an image of red pepper plant from the feild our system will count the total number of the fruit, and calculate the precentage of the ripe fruits.
                #### \n
                #### We hope you'll enjoy it


                ''') 

# Run image
if app_mode == 'Analyse a plant':
    
    
    st.sidebar.markdown('---') # adds a devider (a line)
    
    # side bar
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] . div:first-child{
            width: 350px
        }

        [data-testid="stSidebar"][aria-expanded="false"] . div:first-child{
            width: 350px
            margin-left: -350px
        }    
        </style>

        """,
        unsafe_allow_html=True,


    )


    # read an image from the user
    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    # assign the uplodaed image from the buffer, by reading it in
    if img_file_buffer is not None:
        image = io.imread(img_file_buffer)
    else: # if no image was uploaded, then segment the demo image
        demo_path_img="demo_img.jpg"
        image = io.imread(demo_path_img)

    # display on the sidebar the uploaded image
    st.sidebar.text('Original Image')
    st.sidebar.image(image)
    
   
    
    # Display the result on the right (main frame)
    st.subheader('Output Image')
    st.image(image, use_column_width=True)
    
    
    
    
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://c8.alamy.com/zooms/9/199f27fff05e4797a5d81ed431312661/2ddh245.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

# ----- The img analysis code -----

import libs
import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
import Mask_RCNN.mrcnn.config as Config
import Mask_RCNN.mrcnn.utils as utils
import Mask_RCNN.mrcnn as visualize
import Mask_RCNN.mrcnn.model as modellib

ROOT_DIR = 'Mask_RCNN'

sys.path.append(ROOT_DIR) 

class TrainConfig(Config):
    # Give the configuration a recognizable name
    NAME = "redvsgreen"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 5

    LEARNING_RATE = 0.001

    # Number of classes (including background) - IMPORTANT TO CHANGE ACCORDING TO YOUR LABELS IN YOUR JSON
    NUM_CLASSES = 1 + 2  # background + (red + green)

    # All of our training images are 1920x1012
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet50' # resnet50

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 
    
config = TrainConfig()

# getting the models and annotation files

model_filename = 'model.h5'
jsonTrain_filename = 'train.json'

class InferenceConfig(TrainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.65 # CHANGE HERE IF YOU WANT

inference_config = InferenceConfig()

# Recreate the model in inference mode
test_model = modellib.MaskRCNN(
    mode="inference", 
    config=inference_config,
    model_dir='./')

model_path = f'./{model_filename}'
print(model_path)

test_model.load_weights(model_path, by_name=True)

class InferenceConfig(TrainConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.65 # CHANGE HERE IF YOU WANT

inference_config = InferenceConfig()

# Recreate the model in inference mode
test_model = modellib.MaskRCNN(
    mode="inference", 
    config=inference_config,
    model_dir='./')

model_path = f'./{model_filename}'
print(model_path)

test_model.load_weights(model_path, by_name=True)



