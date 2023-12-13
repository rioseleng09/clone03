import imageio
import streamlit as st
import numpy as np
from tensorflow.keras.models import model_from_json
import pickle
import mahotas as mh
import imageio  # Add this import statement

# Function to load the model
def load_model():
    # Load model architecture from JSON file
    with open('model.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    # Load the Keras model
    model = model_from_json(loaded_model_json)

    # Load weights into the new model
    model.load_weights('model.h5')

    return model

from PIL import Image
import cv2

# Function to make a diagnosis
def diagnosis(file, model):
    IMM_SIZE = 224  # Replace with your desired size

    # Initialize image variable
    image = None

    try:
        # Attempt to read the image from the file
        pil_image = Image.open(file)
        image = np.array(pil_image)
    except Exception as e:
        # Print an error message if the image cannot be read
        print(f"Error reading image from {file}: {e}")

    # Check if image is None (i.e., an error occurred during image reading)
    if image is None:
        # Handle the error or return an appropriate value
        return None

    # Prepare image for classification
    # Resize the image to the desired size
    resized_image = cv2.resize(image, (IMM_SIZE, IMM_SIZE))

    # Convert RGB to grayscale if the image is in color
    if len(resized_image.shape) > 2:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)

    # Reshape input images
    resized_image = resized_image.reshape(-1, IMM_SIZE, IMM_SIZE, 1)

    # Normalize the data (if needed)
    # You may need to adjust this part based on how you trained your model
    resized_image = resized_image / 255.0

    # Predict the diagnosis
    predicted_probabilities = model.predict(resized_image)
    diag = np.argmax(predicted_probabilities, axis=-1)

    # Load history and lab from pickle files
    with open('history.pickle', 'rb') as f:
        history = pickle.load(f)

    with open('lab.pickle', 'rb') as f:
        lab = pickle.load(f)

    # Find the name of the diagnosis
    diag = list(lab.keys())[list(lab.values()).index(diag[0])]

    return diag


# Main Streamlit app
def main():
    st.title("Chest X-ray Image Diagnosis App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Load the model
        model = load_model()

        try:
            # Get diagnosis
            result = diagnosis(uploaded_file, model)
            st.success(f"The predicted diagnosis is: {result}")
        except Exception as e:
            st.error(f"Error during diagnosis: {e}")
            print("Error during diagnosis:", e)

if __name__ == "__main__":
    main()
