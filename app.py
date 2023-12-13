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

# Function to make a diagnosis
def diagnosis(file, model):
    IMM_SIZE = 224  # Replace with your desired size

    # Initialize image variable
    image = None

    try:
        # Attempt to read the image from the file
        image = imageio.imread(file)
    except Exception as e:
        # Print an error message if the image cannot be read
        print(f"Error reading image from {file}: {e}")

    # Check if image is None (i.e., an error occurred during image reading)
    if image is None:
        # Handle the error or return an appropriate value
        return None

    # Prepare image for classification
    # You may need to adjust this part based on your specific requirements
    # For example, normalizing the image, resizing, etc.

    # Check if the image has more than 2 dimensions (i.e., it's RGB or has an alpha channel)
    if len(image.shape) > 2:
        # Resize RGB and PNG images
        resized_channels = [mh.resize(image[:, :, i], (IMM_SIZE, IMM_SIZE)) for i in range(image.shape[2])]
        image = np.stack(resized_channels, axis=-1)
    else:
        # Resize grayscale images
        image = mh.resize(image, (IMM_SIZE, IMM_SIZE))

    # Convert RGB to grayscale if the image is in color
    if len(image.shape) > 2:
        image = mh.colors.rgb2grey(image[:, :, :3], dtype=np.uint8)

    # Load history and lab from pickle files
    with open('history.pickle', 'rb') as f:
        history = pickle.load(f)

    with open('lab.pickle', 'rb') as f:
        lab = pickle.load(f)

    # Normalize the data (if needed)
    # You may need to adjust this part based on how you trained your model
    image = np.array(image) / 255

    # Reshape input images
    # You may need to adjust this part based on your model's input shape
    image = image.reshape(-1, IMM_SIZE, IMM_SIZE, 1)

    # Predict the diagnosis
    predicted_probabilities = model.predict(image)
    diag = np.argmax(predicted_probabilities, axis=-1)

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
