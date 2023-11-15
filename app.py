# import the rquired libraries.
import numpy as np
import cv2
from keras.models import load_model
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

# Define the emotions.
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

# Load model.
classifier =load_model('model.h5')

# load weights into new model
classifier.load_weights("model_weights.h5")

# Load face using OpenCV
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(0, 255, 255), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_labels[maxindex]
                output = str(finalout)
            label_position = (x, y-10)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

def main():
    st.set_page_config(page_title="Expressify",page_icon="😆",layout="wide" )
    st.title("😆 Expressify: Real-Time Face Emotion Detection")
    st.markdown("<hr>", unsafe_allow_html=True)

    col1,col2=st.columns(2)
    with col1:
        st.title("Introduction:")
        st.write("""
       The Indian education landscape has been undergoing rapid changes for the past ten years owing to the advancement of web-based learning services, specifically eLearning platforms.""")
        st.write("""
Digital platforms might overpower physical classrooms in terms of content quality, but in a
physical classroom, a teacher can see the faces and assess the emotion of the class and tune
their lecture accordingly, whether he is going fast or slow. He can identify students who
need special attention.""")
        st.write("""
While digital platforms have limitations in terms of physical surveillance, it comes with the
power of data and machines, which can work for you.""")
        st.write("""
It provides data in form of video, audio, and texts, which can be analyzed using deep
learning algorithms. A deep learning-backed system not only solves the surveillance issue, but
also removes the human bias from the system, and all information is no longer in the teacher’s
brain but rather translated into numbers that can be analyzed and tracked.
                 """)
    
    with col2:
        st.title("Problem statement:")
        st.write("We aim to solve one of the challenges faced by digital platforms by applying deep learning algorithms to live video data.")
        st.write("We do this by recognizing the facial emotions of the participants using the CNN model.")
        st.title("Dataset used:")
        st.write("""We have utilized the provided from Kaggle :""")
        st.markdown("<a href='https://www.kaggle.com/datasets/msambare/fer2013'>FER 2013 dataset</a>", unsafe_allow_html=True)
        st.write("""
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically
registered so that the face is more or less centered and occupies about the same amount of
space in each image.
""")
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col5,col6=st.columns(2)
    with col5:
        st.title("Model Building and Evaluation:")
        st.subheader("Convolutional Neural Network (CNN):")
        st.write("""A neural network is a way for a computer to process data input. They’re inspired by biological
processes found in human and animal brains. Neural networks are comprised of various layers
of ‘nodes’ or ‘artificial neurons’. Each node processes the input and communicates with the
other nodes.In this way, input filters through the processing of a neural network to create the
output, or answer.""")
        st.write("""Convolutional neural networks were inspired by animal vision. The way the nodes in a CNN
communicate with each other resembles the way some animals see the world.
So, rather than taking everything in as a whole, small areas of an image are taken.
And these small areas overlap to cover the whole image.""")
        
    with col6:
        st.title("What happens inside an CNN?")
        tab1, tab2, tab3 ,tab4,tab5= st.tabs(["Input Layer", "Convolutional Layers", "Rectified Linear Unit (ReLU)","Pooling Layers","Fully Connected Layer"])
        with tab1:
           st.write("An image is fed into the model as an input.")

        with tab2:
           st.write("Instead of looking at the whole picture at once, it scans it in overlapping blocks of pixels. In simple terms, the filters assign a value to the pixels that match them. The more they match, the higher the value.")

        with tab3:
           st.write("ReLU allows for faster and more effective training by mapping negative values to zero and maintaining positive values. This is sometimes referred to as activation, because only the activated features are carried forward into the next layer.")
        with tab4:
            st.write("Pooling simplifies the output by performing nonlinear downsampling, reducing the number of parameters that the network needs to learn.")
            
        with tab5:
            st.write("In a fully connected layer, every node receives the input from every node in the previous layer. This is where all the features extracted by the convolutional neural network get combined.")  
            
    st.markdown("<hr>", unsafe_allow_html=True)
   
    col3,col4=st.columns(2)
    with col3:
        st.header("Webcam Live Feed")
        st.subheader('''
        Welcome to the other side of the SCREEN!!!
        * Get ready with all the emotions you can express. 
        ''')
        st.write("1. Click Start to open your camera and give permission for prediction")
        st.write("2. This will predict your emotion.")
        st.write("3. When you done, click stop to end.")
    with col4:
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col7,col8= st.columns(2)
    with col7:
        st.subheader('👤 Profile')
        st.write('I possess a deep love for coding and a strong desire to continuously improve my skills. With a demonstrated proficiency in multiple programming languages and a natural curiosity for problem-solving, I am eager to contribute my technical expertise to any organization')
        st.markdown("<p style= font-size:16px;>Deepraj Bera</p><p style= font-size:12px;>Full Stack | ML</p><a href='https://github.com/deepraj21'><img src='https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white' alt='GitHub'></a>", unsafe_allow_html=True)
    with col8:
        st.subheader('✉️ Find Me')
        st.markdown("<a href='mailto: deepraj21.bera@gmail.com' style='text-decoration:none;'>deepraj21.bera@gmail.com</a>",unsafe_allow_html=True)
        st.markdown("<a href='mailto: 21051302@kiit.ac.in'>21051302@kiit.ac.in</a>",unsafe_allow_html=True)


    st.markdown("<center><p>© 2023 Expressify</p><center>",unsafe_allow_html=True)


if __name__ == "__main__":
    main()