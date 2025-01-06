import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the pre-trained model
model = load_model('model.keras')

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear",
                  "Happy", "Sad", "Surprise", "Neutral"]

# Streamlit app layout
st.title("Emotion Classification App")
st.write("Upload an image to classify the emotion.")

# Function to preprocess the image


def preprocess_image(image):
    """
    Preprocess the input image for emotion classification.

    Parameters:
    image (PIL.Image): The input image to preprocess.

    Returns:
    np.ndarray: The preprocessed image ready for model prediction.
    """
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((48, 48))  # Resize to 48x48
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to predict emotion


def predict_emotion(image_array):
    """
    Predict the emotion from the preprocessed image.

    Parameters:
    image_array (np.ndarray): The preprocessed image array.

    Returns:
    str: The predicted emotion label.
    """
    prediction = model.predict(image_array)
    predicted_emotion = emotion_labels[np.argmax(prediction)]
    return predicted_emotion


# Initialize session state for clearing
if 'uploaded_file' not in st.session_state:
    st.session_state['uploaded_file'] = None

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

# Clear button
if st.button("Clear"):
    st.session_state['uploaded_file'] = None
    st.experimental_rerun()

# Display and process the uploaded image
if uploaded_file is not None:
    st.session_state['uploaded_file'] = uploaded_file

if st.session_state['uploaded_file'] is not None:
    # Convert uploaded file to an image
    img = Image.open(st.session_state['uploaded_file'])
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Make prediction
    predicted_emotion = predict_emotion(img_array)

    # Display prediction
    st.write(f"Predicted Emotion Is: **{predicted_emotion}**")
