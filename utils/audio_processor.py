import librosa
import numpy as np
import json
import os


def filepath():
    if os.path.isfile('config.json'):
        path = 'config.json'
    else:
        path = './../config.json'
    return path


def load_config(path=filepath()):
    with open(path, 'r') as file:
        return json.load(file)


config = load_config()


def load_audio_file(file_path):
    """
    Loads the audio file and returns the audio data and sample rate.
    """
    audio, sample_rate = librosa.load(file_path, sr=config['audio']['sample_rate'])
    return audio, sample_rate


def extract_features(audio, sample_rate):
    """
    Extracts 13 MFCCs from the audio data.
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    if mfccs_processed.shape[0] != 13:
        raise ValueError("Number of MFCCs extracted does not match the model's expected input size.")
    return mfccs_processed


def process_audio_file(audio_path):
    """
    Loads the audio file, extracts features, and prints them.
    """
    audio, sample_rate = load_audio_file(audio_path)
    features = extract_features(audio, sample_rate)
    print("Extracted Features:", features)


def get_filenames_in_directory(directory_path):
    """
        Retrieves the filenames in a given directory, excluding subdirectories. This function aids in listing the outputs generated, such as plots or logs.
        """
    try:
        filenames = os.listdir(directory_path)
        filenames = [f for f in filenames if os.path.isfile(os.path.join(directory_path, f))]
        return filenames
    except FileNotFoundError:
        print(f"The directory {directory_path} was not found.")
        return []


if __name__ == "__main__":
    # Process audio files using paths from the configuration
    human_audio_path = config['audio']['human_audio_path']
    ai_audio_path = config['audio']['ai_audio_path']
    process_audio_file(human_audio_path)
    process_audio_file(ai_audio_path)
