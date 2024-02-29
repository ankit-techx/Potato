
# Potato Disease Classification

This project aims to classify potato diseases(healthy, early blight, late blight) using machine learning and deep learning techniques. It includes code for training a convolutional neural network (CNN) model using TensorFlow/Keras and integrating the trained model into a web application using Streamlit.

## Dataset

The dataset used for training the model is the "PlantVillage" dataset, which contains images of various plant diseases, including potato diseases. The dataset can be obtained from [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset).

## Requirements

To run the training code and the Streamlit web application, the following dependencies are required:

- Python 3.x
- TensorFlow
- Keras
- Streamlit
- NumPy
- Matplotlib
- Pillow
- Requests

## Training the Model

To train the model, run the `potato_training.ipynb` notebook. Ensure that the dataset directory is correctly specified in the notebook.

The notebook performs the following steps:
1. Data augmentation and preprocessing.
2. Loading the pre-trained VGG16 model.
3. Freezing the layers of the pre-trained model.
4. Adding custom classification layers.
5. Compiling and training the model.
6. Evaluating the model.
7. Saving the trained model as `potato_disease_classification_model.h5`.

## Running the Web Application

To run the Streamlit web application, execute the `app.py` script. This application allows users to upload an image or provide a URL to classify potato diseases.

```bash
streamlit run app.py
```

The web application loads the trained model and provides a simple interface for users to interact with.

## Usage

- Upload Image: Allows users to upload an image from their local machine for classification.
- Provide Image URL: Enables users to provide a URL of an image hosted online for classification.

## Acknowledgments

- The dataset used in this project is provided by [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset).
- The pre-trained VGG16 model is obtained from the Keras Applications module.

![image](https://github.com/ankit-techx/Potato_leaves_disease_classification/assets/101319910/c8fef0bb-be24-4fd1-af1d-f30f5dfcd7fd)

