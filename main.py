import os
import librosa
import pickle
import sklearn
import numpy as np
import streamlit as st

# Function to load the audio model
@st.cache_resource
def load_model():
    with open('classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        save_dir = "audio"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

# Function to apply pre-emphasis filter
def pre_emphasis(audio, pre_emphasis_coeff=0.97):
    return np.append(audio[0], audio[1:] - pre_emphasis_coeff * audio[:-1])

# Function to extract audio features
# Fungsi untuk ekstraksi MFCC dari file audio
def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, sr=None)
        print(f'sample rate: {sample_rate}')
        if len(audio) > 4 * sample_rate:
            audio = audio[:4 * sample_rate]
        # Pad audio to 4 seconds if shorter than 4 seconds
        elif len(audio) < 4 * sample_rate:
            padding = 4 * sample_rate - len(audio)
            audio = np.pad(audio, (0, padding), 'constant')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        print(f'mfccs_mean: {mfccs_mean}')
    except Exception as error:
        print(f"Error encountered while parsing file: {file_name}")
        print(f'error : {error}')
        return None
    return mfccs_mean
# Mapping numeric labels to original or descriptive emotion labels
label_mapping = {
    0: 'neutral',  # Adjust these according to your actual labels
    1: 'calm',
    2: 'happy',
    3: 'sad',
    4: 'angry',
    5: 'fearful',
    6: 'disgust',
    7: 'surprised'
}

# Function to predict audio emotion
def predict_audio(path, model):
    feature = extract_features(path)
    if feature is not None:
        feature_array = np.array(feature)
        feature_to_predict = feature_array.reshape(1, -1)
        prediction = model.predict(feature_to_predict)
        if prediction:
            predicted_label = label_mapping.get(prediction[0], "Unknown")  # Translate to the mapped label
            return predicted_label
    return None

# Main Streamlit interface
def main():
    st.title("Deteksi Emosi dari Audio")
    st.write("Upload file audio dan sistem akan memprediksi emosi dalam file audio tersebut.")

    uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        path = save_uploaded_file(uploaded_file)
        model = load_model()
        prediction = predict_audio(path, model)
        if prediction is not None:
            st.write(f"Prediksi: {prediction}")
        if os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    main()
