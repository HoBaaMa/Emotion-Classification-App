# Emotion Classification App

Welcome to the Emotion Classification App! This application uses a pre-trained deep learning model to classify emotions from uploaded images.

## Features

- **Emotion Detection**: Upload an image, and the app will classify the emotion displayed.
- **User-Friendly Interface**: Built with Streamlit for an intuitive and interactive user experience.

## Technologies Used

- **Python**: Core programming language for development.
- **TensorFlow/Keras**: For loading and using the pre-trained emotion classification model.
- **Streamlit**: To create an interactive web interface.
- **PIL (Python Imaging Library)**: For image processing.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/HoBaaMa/Emotion-Classification-App.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Emotion-Classification-App
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Open your web browser and go to `http://localhost:8501` to access the app.
3. Upload an image to classify the emotion.

## How It Works

- The app preprocesses the uploaded image by converting it to grayscale, resizing it to 48x48 pixels, and normalizing the pixel values.
- The preprocessed image is then fed into a pre-trained model to predict the emotion.
- The predicted emotion is displayed on the app interface.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Special thanks to my mentor, Ahmed Hikal, AMIT Learning and ODC for their support and guidance throughout this project.
