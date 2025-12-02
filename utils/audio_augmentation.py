import os
import librosa
import soundfile as sf
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


config = load_config()['audio_augmentation']


def change_pitch_and_speed(audio, sample_rate, pitch_factor, speed_factor):
    """
    Change the pitch and speed of an audio signal.
    """
    audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_factor)
    audio = librosa.effects.time_stretch(audio, rate=speed_factor)
    return audio


def augment_audio(directory_path, output_path_prefix, sample_rate):
    """
    Augment each audio file in the given directory by changing its pitch and speed,
    and save the augmented files.
    """
    # Ensure the output directory exists
    os.makedirs(output_path_prefix, exist_ok=True)

    # Iterate through each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):  # Check if it's an audio file
            file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {file_path}")

            # Load the audio file
            audio, sr = librosa.load(file_path, sr=sample_rate)

            # Process the audio file with each combination of pitch and speed factors
            for pitch_factor in config['pitch_factors']:
                for speed_factor in config['speed_factors']:
                    augmented_audio = change_pitch_and_speed(audio, sr, pitch_factor, speed_factor)
                    output_file_name = f"{filename[:-4]}_pitch{pitch_factor}_speed{speed_factor}.wav"
                    output_file_path = os.path.join(output_path_prefix, output_file_name)
                    sf.write(output_file_path, augmented_audio, sr)
                    print(f"Augmented file saved to: {output_file_path}")
        else:
            print(f"Skipping non-audio file: {filename}")


if __name__ == "__main__":
    # Use paths from config
    augment_audio(config['paths']['original_human_voice_dir'], config['paths']['output_human_voice_dir'],
                  config['sample_rate'])
    augment_audio(config['paths']['original_ai_voice_dir'], config['paths']['output_ai_voice_dir'],
                  config['sample_rate'])
