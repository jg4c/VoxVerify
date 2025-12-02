import torch
import torch.nn.functional as F
import time
import os
import soundfile as sf
from capture_audio import capture_audio
from utils.audio_processor import extract_features
from model.voice_classifier import VoiceClassifier
import json


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


def load_model(model_path):
    """
    Load the pre-trained model from the specified path.
    """
    model = VoiceClassifier()
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device(config['model']['device'])))  # Use config for device
    model.eval()
    return model


def analyse_audio(audio_data, model, sample_rate):
    """
    Analyze the captured audio data using the pre-trained model.
    """
    # Extract features and convert to the right shape
    features = extract_features(audio_data, sample_rate)
    features = torch.tensor([features], dtype=torch.float)

    with torch.no_grad():
        logits = model(features)
        probabilities = F.softmax(logits, dim=1)

        # Index 1 is 'AI' and 0 is 'Human' and logits are structured [human_score, ai_score]
        is_human_prob = probabilities[0, 0].item()  # Probability of 'Human'

        # If the probability of 'Human' is greater than 0.5, label as 'Human', otherwise 'AI'
        result_text = "Human" if is_human_prob > 0.5 else "AI"
        confidence = is_human_prob if result_text == "Human" else 1 - is_human_prob

    return result_text, confidence * 100  # Convert to percentage


def main():
    """
    Main function to handle real-time audio capture and analysis.
    """
    model_path = config['model']['path']  # config for model path
    audio_format = config['audio']['format']  # config for audio format
    directory = config['audio']['directory']  # config for directory

    model = load_model(model_path)  # Load model with configured path

    try:
        while True:
            print("Please speak into the microphone.")
            audio_data = capture_audio(duration=5, fs=config['audio']['sample_rate'])  # config for sample rate
            prediction = analyse_audio(audio_data, model)
            print(f"Prediction: {prediction}")

            # Save the audio data
            file_name = os.path.join(directory, f"audio_{int(time.time())}.{audio_format.lower()}")
            sf.write(file_name, audio_data, config['audio']['sample_rate'])  # config for sample rate
            print(f"Audio saved to {file_name}")

    except KeyboardInterrupt:
        print("Analysis stopped.")


if __name__ == "__main__":
    main()
