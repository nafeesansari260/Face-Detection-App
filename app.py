import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Face Detection System",
    page_icon="😎",
    layout="wide"
)


# ---------------- LOAD HAAR CASCADE ----------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_smile.xml"
)



# ---------------- FUNCTIONS ----------------


def convert_image(image):

    img = np.array(
        image.convert("RGB")
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )

    return img



# FACE DETECTION

def detect_faces(image):

    img = convert_image(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )


    for (x,y,w,h) in faces:

        cv2.rectangle(
            img,
            (x,y),
            (x+w,y+h),
            (255,0,0),
            3
        )

        cv2.putText(
            img,
            "Face",
            (x,y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,0,0),
            2
        )


    return img, faces




# EYE DETECTION

def detect_eyes(image):

    img = convert_image(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )


    eyes = eye_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )


    for (x,y,w,h) in eyes:

        cv2.rectangle(
            img,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )


    return img, eyes





# SMILE DETECTION

def detect_smiles(image):

    img = convert_image(image)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )


    smiles = smile_cascade.detectMultiScale(
        gray,
        1.1,
        4
    )


    for (x,y,w,h) in smiles:

        cv2.rectangle(
            img,
            (x,y),
            (x+w,y+h),
            (0,0,255),
            2
        )


    return img, smiles





# CARTOON EFFECT

def cartoonize_image(image):

    img = convert_image(image)


    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )


    gray = cv2.medianBlur(
        gray,
        5
    )


    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        9
    )


    color = cv2.bilateralFilter(
        img,
        9,
        300,
        300
    )


    cartoon = cv2.bitwise_and(
        color,
        color,
        mask=edges
    )


    return cartoon





# CANNY EDGE

def canny_edge(image):

    img = convert_image(image)


    blur = cv2.GaussianBlur(
        img,
        (11,11),
        0
    )


    edges = cv2.Canny(
        blur,
        100,
        150
    )


    return edges




# ---------------- MAIN APP ----------------


def main():

    st.title(
        "😎 AI Face Detection System"
    )


    st.write(
        "Built with Streamlit + OpenCV"
    )


    menu = st.sidebar.selectbox(
        "Select Option",
        [
            "Detection",
            "About"
        ]
    )



    if menu=="Detection":


        st.subheader(
            "Upload Image"
        )


        image_file = st.file_uploader(
            "Choose Image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )



        if image_file:


            image = Image.open(
                image_file
            )


            col1,col2 = st.columns(2)


            with col1:

                st.write(
                    "Original Image"
                )

                st.image(
                    image,
                    width=300
                )



            option = st.sidebar.radio(
                "Image Enhancement",
                [
                    "Original",
                    "Gray Scale",
                    "Contrast",
                    "Brightness",
                    "Blur"
                ]
            )



            if option=="Gray Scale":

                img = np.array(image)

                gray = cv2.cvtColor(
                    img,
                    cv2.COLOR_RGB2GRAY
                )

                st.image(gray)



            elif option=="Contrast":

                value = st.sidebar.slider(
                    "Contrast",
                    0.5,
                    3.5,
                    1.0
                )


                enhancer = ImageEnhance.Contrast(
                    image
                )

                st.image(
                    enhancer.enhance(value)
                )



            elif option=="Brightness":

                value = st.sidebar.slider(
                    "Brightness",
                    0.5,
                    3.5,
                    1.0
                )


                enhancer = ImageEnhance.Brightness(
                    image
                )


                st.image(
                    enhancer.enhance(value)
                )



            elif option=="Blur":

                img = convert_image(
                    image
                )

                blur = cv2.GaussianBlur(
                    img,
                    (11,11),
                    0
                )


                st.image(
                    blur
                )



            task = st.sidebar.selectbox(
                "Detection Type",
                [
                    "Face",
                    "Eyes",
                    "Smile",
                    "Cartoon",
                    "Canny"
                ]
            )



            if st.button(
                "Process"
            ):


                if task=="Face":

                    result,count = detect_faces(
                        image
                    )


                    st.image(
                        result,
                        channels="BGR"
                    )


                    st.success(
                        f"{len(count)} Face Found"
                    )



                elif task=="Eyes":

                    result,count = detect_eyes(
                        image
                    )


                    st.image(
                        result,
                        channels="BGR"
                    )


                    st.success(
                        f"{len(count)} Eyes Found"
                    )



                elif task=="Smile":

                    result,count = detect_smiles(
                        image
                    )


                    st.image(
                        result,
                        channels="BGR"
                    )


                    st.success(
                        f"{len(count)} Smile Found"
                    )



                elif task=="Cartoon":

                    result = cartoonize_image(
                        image
                    )


                    st.image(
                        result,
                        channels="BGR"
                    )



                elif task=="Canny":

                    result = canny_edge(
                        image
                    )


                    st.image(
                        result
                    )



    else:

        st.subheader(
            "About Project"
        )


        st.write(
            """
            AI Face Detection System

            Technologies:
            - Python
            - Streamlit
            - OpenCV
            - Haar Cascade

            Features:
            ✔ Face Detection
            ✔ Eye Detection
            ✔ Smile Detection
            ✔ Cartoon Effect
            ✔ Edge Detection
            """
        )



if __name__=="__main__":
    main()