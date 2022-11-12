import streamlit as st
#import joblib
from keras.preprocessing import image
from io import BytesIO


import cv2
import gdown
import glob
import numpy as np
import os
import patoolib

from pyngrok import ngrok
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ngrok_process = ngrok.get_ngrok_process()
DATA_ROOT = '/content'
# os.makedirs(DATA_ROOT, exist_ok=True)
max_samples = 3000

blood_slide_url = 'https://drive.google.com/uc?id=1lffxAG8gykh1dh1pCP34uRkH3XMwuNt-'
blood_slide_path = os.path.join(DATA_ROOT, 'blood_slide.jpg')
gdown.download(blood_slide_url, blood_slide_path, True)

malaria_imgs_path = os.path.join(DATA_ROOT, './malaria_images')

print("Downloaded Data")

u_malaria_img_paths = glob.glob('./malaria_images/Uninfected/*.png')
p_malaria_img_paths = glob.glob('./malaria_images/Parasitized/*.png')

NUM_SAMPLES = len(u_malaria_img_paths) + len(p_malaria_img_paths)

X = []
y = []

X_g = []

for i in tqdm(range(max_samples)):
  img = cv2.imread(u_malaria_img_paths[i])
  X.append(cv2.resize(img,(50,50)))

  gray_img = cv2.imread(u_malaria_img_paths[i],0)
  X_g.append(cv2.resize(gray_img,(50,50)))

  y.append(0)

for i in tqdm(range(max_samples)):
  img = cv2.imread(p_malaria_img_paths[i])
  X.append(cv2.resize(img,(50,50)))

  gray_img = cv2.imread(p_malaria_img_paths[i],0)
  X_g.append(cv2.resize(gray_img,(50,50)))

  y.append(1)

X = np.stack(X)
X_g = np.stack(X_g)
X_reshaped = np.reshape(X_g,(X_g.shape[0],2500))

y = np.array(y)

# blood_samples_dir = 'blood_samples'
# if (os.path.exists(blood_samples_dir) == False):
#   os.mkdir(blood_samples_dir)

# for i, img in enumerate(X[2995:3005]):
#   plt.imsave('test_img_{}.jpg'.format(i), img)
  
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=.33, random_state=4)

model = SVC()
model.fit(X_train, y_train)

st.image('malaria.jpg')
st.markdown("## Health Care at Your Fingertips")
st.markdown("### Malaria Detection with Machine Learning")
st.markdown("""
Globally, billions of people own smartphones, which means that ML apps can help diagnose and treat large numbers of people!

Infecting over 200 million people every year, malaria kills more than 200,000 children, making it one of the world's most deadly diseases. Using this app, we can help fight malaria, one of the world's most deadly diseases. With this tool, healthcare workers in rural or developing areas can diagnose malaria without expensive laboratory equipment. We will use computer vision to detect malaria parasites in blood cells uploaded under a microscope by healthcare workers.

**Made by Athulya Saravanakumar**
""")
st.markdown("""Example of Parasitized Blood Sample""")
st.image("./malaria_images/Parasitized/infected.png")
st.markdown("""Example of Uninfected Blood Cell Sample""")
st.image("./malaria_images/Uninfected/good.png")
#Name of Classes


#Uploading the blood cell image
object_image = st.file_uploader("""Upload Blood Cell Sample""", type=['png','jpg','webp','jpeg'])
submit = st.button('Diagnose')

#On predict button click
if submit:

    if object_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(object_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        # st.image(opencv_image, channels="BGR")
        # print(opencv_image.shape[0], opencv_image[1])        
        small = cv2.resize(opencv_image, (50,50))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        new_image = np.reshape(gray,(1,2500))

        # opencv_image.shape = (1,2500)

        # opencv_image.shape = (1,224,224,3)
        # small = cv2.resize(opencv_image, (50,50)) 
        # gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # np.reshape(opencv_image,(15,2645))
        
        predictions = model.predict(new_image)
        if(predictions[0]==0):
            predicted_classes = "no malaria!"
        else:
            predicted_classes = "you have malaria!"


        # Displaying the image
        #st.image(object_image, channels="BGR")
        st.markdown("Diagnosis: " +  predicted_classes)