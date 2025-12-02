📌 Features
🎤 Real-Time Voice Recording

Running app.py launches the application interface, where you will see two buttons:

Start Recording

Stop Recording

These allow you to capture live audio directly from your microphone.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

📁 Audio File Upload

You may also drag and drop an audio file into the application.
After uploading, click Analyse Audio to process the file.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

🧠 Deepfake Detection Result

After recording or analysing an audio clip, the system will display a prediction score showing whether the voice is:

Human, or

AI-generated

🛠 Installation
Recommended Python Version

Python 3.9

Install Required Packages

Run:

pip install -r requirements.txt

🔧 Model Training & Audio Augmentation
Retrain the Deepfake Voice Classifier

To retrain the model, run:

python Voice_Classifier.py

Generate Augmented Training Data

To augment your audio dataset, run:

python Audio_Augmentation.py
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

🚀 Getting Started

Install dependencies

Run the application:

python app.py


Choose to:

record live audio, or

drag & drop an audio file

View the prediction score to see whether the voice is Human or AI.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

🙌 Thank You

Thank you for using the Real-Time Deepfake Voice Detection and Verification application.
Enjoy exploring and experimenting with the system!

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

⚖️ License & Rights

© 2025 Jamal Gwarada — All Rights Reserved.

This project and all associated files are the intellectual property of the author.
No part of this repository may be copied, distributed, modified, or used for commercial or non-commercial purposes without explicit written permission from the owner.
