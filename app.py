import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Face Detection System",
    page_icon="😎",
    layout="wide"
)



# ---------------- LOAD MODELS ----------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models"
)



def load_cascade(filename):

    path = os.path.join(
        MODEL_PATH,
        filename
    )

    if not os.path.exists(path):
        st.error(
            f"Missing file: {filename}"
        )
        st.stop()


    cascade = cv2.CascadeClassifier(path)


    if cascade.empty():
        st.error(
            f"Cannot load {filename}"
        )
        st.stop()


    return cascade




face_cascade = load_cascade(
    "haarcascade_frontalface_default.xml"
)


eye_cascade = load_cascade(
    "haarcascade_eye.xml"
)


smile_cascade = load_cascade(
    "haarcascade_smile.xml"
)




# ---------------- IMAGE CONVERSION ----------------


def convert_image(image):

    img = np.array(
        image.convert("RGB")
    )


    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )


    return img





# ---------------- FACE DETECTION ----------------


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


    for x,y,w,h in faces:

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





# ---------------- EYE DETECTION ----------------


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


    for x,y,w,h in eyes:

        cv2.rectangle(
            img,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )


    return img, eyes





# ---------------- SMILE DETECTION ----------------


def detect_smile(image):

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


    for x,y,w,h in smiles:

        cv2.rectangle(
            img,
            (x,y),
            (x+w,y+h),
            (0,0,255),
            2
        )


    return img, smiles





# ---------------- CARTOON ----------------


def cartoon_effect(image):

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





# ---------------- CANNY ----------------


def canny(image):

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





# ---------------- MAIN ----------------


def main():

    st.title(
        "😎 AI Face Detection System"
    )


    st.write(
        "Streamlit + OpenCV + Haar Cascade"
    )



    menu = st.sidebar.selectbox(
        "Menu",
        [
            "Detection",
            "About"
        ]
    )



    if menu=="Detection":


        file = st.file_uploader(
            "Upload Image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )


        if file:


            image = Image.open(
                file
            )


            st.image(
                image,
                caption="Original Image",
                width=350
            )



            task = st.sidebar.selectbox(
                "Choose Feature",
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
                        f"{len(count)} Face Detected"
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
                        f"{len(count)} Eyes Detected"
                    )



                elif task=="Smile":

                    result,count = detect_smile(
                        image
                    )


                    st.image(
                        result,
                        channels="BGR"
                    )


                    st.success(
                        f"{len(count)} Smile Detected"
                    )



                elif task=="Cartoon":

                    result = cartoon_effect(
                        image
                    )


                    st.image(
                        result,
                        channels="BGR"
                    )



                elif task=="Canny":

                    result = canny(
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
            - OpenCV
            - Streamlit
            - Haar Cascade

            Features:
            ✔ Face Detection
            ✔ Eye Detection
            ✔ Smile Detection
            ✔ Cartoon Filter
            ✔ Edge Detection
            """
        )





if __name__=="__main__":
    main()