import streamlit as st
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import pickle

# Function to load the model
def load_model(model_path='model.json', weights_path='model.h5'):
    # Load model architecture from JSON file
    with open(model_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    # Close the JSON file
    model = model_from_json(loaded_model_json)

    # Load weights into the new model
    model.load_weights(weights_path)

    return model

# Function to make a diagnosis
def diagnosis(file, model, IMM_SIZE):
    # Load and preprocess the image
    img = image.load_img(file, target_size=(IMM_SIZE, IMM_SIZE), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]

    # Predict the diagnosis
    predicted_probabilities = model.predict(img_array)
    predicted_class = np.argmax(predicted_probabilities, axis=-1)[0]

    # Map the predicted class to the diagnosis
    diagnosis_mapping = {0: "Covid", 1: "Normal", 2: "Pneumonia"}
    predicted_diagnosis = diagnosis_mapping[predicted_class]

    return predicted_diagnosis

# Main Streamlit app
def main():
    st.title("Chest X-ray Image Diagnosis App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # File upload for lab and history pickles
    lab_pickle_file = st.file_uploader("Upload lab.pickle file", type=["pickle"])
    history_pickle_file = st.file_uploader("Upload history.pickle file", type=["pickle"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Load the model
        model = load_model()

        # Specify the image size
        IMM_SIZE = 224

        try:
            # Get diagnosis
            result = diagnosis(uploaded_file, model, IMM_SIZE)
            st.success(f"The predicted diagnosis is: {result}")
        except Exception as e:
            st.error(f"Error during diagnosis: {e}")
            print("Error during diagnosis:", e)

        # Additional processing using lab and history pickles
        if lab_pickle_file is not None and history_pickle_file is not None:
            try:
                # Load lab data
                with open(lab_pickle_file, 'rb') as lab_file:
                    lab_data = pickle.load(lab_file)

                # Load history data
                with open(history_pickle_file, 'rb') as history_file:
                    history_data = pickle.load(history_file)

                # Perform further analysis using lab and history data
                st.write("Lab Data:", lab_data)
                st.write("History Data:", history_data)
                # ...

            except Exception as e:
                st.error(f"Error during additional processing: {e}")
                print("Error during additional processing:", e)

# ...

if __name__ == "__main__":
    main()
