# Import all the necessary libraries
import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from keras.preprocessing import image
from keras.models import load_model

# Load background image
background_image = Image.open("potato.webp")
st.image(background_image, use_column_width=True)

# Load the trained model
model = load_model("potato_disease_classification_model.h5")

# Define constants
IMAGE_SIZE = 256

def preprocess_image(img):
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array /= 255.0  # Rescale pixel values to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image_class(img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    return prediction[0]  # Return probabilities

def get_class_label(class_index):
    labels = ["Early Blight", "Healthy", "Late Blight"]
    return labels[class_index]

# Streamlit app
st.title("Potato Disease Classification")

# Option for user to choose between uploading an image or providing URL
option = st.radio("Select Input Option:", ("Upload Image", "Provide Image URL"))

if option == "Upload Image":
    # File uploader for user input
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Predict the class of the uploaded image
        prediction = predict_image_class(img)
        predicted_class_index = np.argmax(prediction)
        predicted_class_label = get_class_label(predicted_class_index)
        predicted_class_percentage = round(prediction[predicted_class_index] * 100, 2)
        st.write("Predicted Class:", predicted_class_label)
        st.write("Predicted percent:", f"{predicted_class_percentage}%")

elif option == "Provide Image URL":
    # Text input for image URL
    image_url = st.text_input("Enter Image URL:")

    # Button to trigger image classification
    classify_button = st.button("Classify")

    if classify_button:
        if image_url:
            try:
                # Download the image from the URL
                response = requests.get(image_url)
                response.raise_for_status()  # Raise HTTPError for bad responses
                img = Image.open(BytesIO(response.content))
                st.image(img, caption="Image from URL", use_column_width=True)

                # Predict the class of the image from URL
                prediction = predict_image_class(img)
                predicted_class_index = np.argmax(prediction)
                predicted_class_label = get_class_label(predicted_class_index)
                predicted_class_percentage = round(prediction[predicted_class_index] * 100, 2)
                st.write("Predicted Class:", predicted_class_label)
                st.write("Predicted percent:", f"{predicted_class_percentage}%")
            except requests.exceptions.RequestException as e:
                st.write("Error: Failed to retrieve image from URL.")
            except Exception as e:
                st.write("Error:", e)
