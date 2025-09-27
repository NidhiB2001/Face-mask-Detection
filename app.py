''' streamlit UI for image detection, to run command>> streamlit run app.py '''


import streamlit as st
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model(r"best_tl_model_keras.h5")  # Change path 

# Class labels
class_labels = ['with_mask', 'without_mask']

# MTCNN face detector
detector = MTCNN()

# Preprocess face for model
def preprocess_face(face_img):
    face = cv2.resize(face_img, (224, 224))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# Predict mask or not
def predict_face(face_img):
    processed = preprocess_face(face_img)
    pred = model.predict(processed)
    class_id = np.argmax(pred)
    confidence = pred[0][class_id]
    return class_labels[class_id], confidence

# Detect and annotate image
def detect_faces(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_image)

    for face in faces:
        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)

        face_img = image[y:y+h, x:x+w]
        if face_img.shape[0] == 0 or face_img.shape[1] == 0:
            continue

        label, conf = predict_face(face_img)

        if conf >= 0.98:
            color = (0, 255, 0) if label == 'with_mask' else (0, 0, 255)
            label_text = f"{label}: {conf:.2f}"
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return image

# Streamlit UI
st.title("😷 Face Mask Detection")
st.write("Upload an image to check if faces have masks or not.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

        result_img = detect_faces(img)

        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)
    except:
        st.write("🤔 Please take another image!")






# '''live detection'''


# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np
# from mtcnn.mtcnn import MTCNN
# from tensorflow.keras.models import load_model


# # Load trained model (custom CNN or MobileNetV2)
# model = load_model(r"best_tl_model_keras.h5")  # change model path

# # Class labels (adjust based on your training labels)
# class_labels = ['with_mask', 'without_mask']

# detector = MTCNN()

# def preprocess_face(face_img):
#     face = cv2.resize(face_img, (224, 224))
#     face = face.astype('float32') / 255.0
#     face = np.expand_dims(face, axis=0)
#     return face

# def predict_face(face_img):
#     processed = preprocess_face(face_img)
#     pred = model.predict(processed)
#     class_id = np.argmax(pred)
#     confidence = pred[0][class_id]
#     return class_labels[class_id], confidence




# def detect_from_webcam():
#     cap = cv2.VideoCapture(0)  # 0 for default webcam

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = detector.detect_faces(rgb_frame)

#         for face in faces:
#             x, y, w, h = face['box']
#             x, y = max(0, x), max(0, y)

#             face_img = frame[y:y+h, x:x+w]
#             if face_img.shape[0] == 0 or face_img.shape[1] == 0:
#                 continue

#             label, conf = predict_face(face_img)
#             if conf >= 0.95:
#                 color = (0, 255, 0) if label == 'with_mask' else (0, 0, 255)
#                 label_text = f"{label}: {conf:.2f}"

#                 cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
#                 cv2.putText(frame, label_text, (x, y - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#         cv2.imshow("Mask Detection - Webcam", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()




# detect_from_webcam()

