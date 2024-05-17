from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from keras.metrics import AUC
import numpy as np
import pandas as pd
import os
import librosa
import librosa, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from pydub import AudioSegment
from werkzeug.utils import secure_filename

app = Flask(__name__)

final = pd.read_pickle("extracted_df.pkl")
y = np.array(final["name"].tolist())
le = LabelEncoder()
le.fit_transform(y)
Model1_ANN = load_model("Model1.h5")

# Define a dictionary to map bird species names to their corresponding image paths
bird_images = {
    'Song Sparrow': 'static/DATASET/photos/Song Sparrow.jpg',
    'Northern Mockingbird': 'static/DATASET/photos/Northern Mockingbird.jpg',
    'Northern Cardinal': 'static/DATASET/photos/Northern Cardinal.jpg',
    'American Robin': 'static/DATASET/photos/American Robin.jpg',
    'Bewick\'s Wren': 'static/DATASET/photos/Bewicks Wren.jpg',
    'Common Cuckoo': 'static/DATASET/photos/Common Cuckoo.jpg',
    'Eurasian Bluetit': 'static/DATASET/photos/Eurasian Bluetit.jpg',
    'European Goldfinch': 'static/DATASET/photos/European Goldfinch.jpg',
    'House Sparrow': 'static/DATASET/photos/House Sparrow.jpg',
    'Eurasian Jay': 'static/DATASET/photos/Eurasian Jay.jpg'
    # Add more bird species if needed
}

def extract_feature(audio_path):
    audio_data, sample_rate = librosa.load(audio_path, res_type="kaiser_fast")
    feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    feature_scaled = np.mean(feature.T, axis=0)
    return np.array([feature_scaled])

def ANN_print_prediction(audio_path):
    prediction_feature = extract_feature(audio_path)
    predicted_vector = np.argmax(Model1_ANN.predict(prediction_feature), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    image_path = bird_images.get(predicted_class[0], 'static/DATASET/photos/default_image.jpg')
    return predicted_class[0], image_path

@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')
    
@app.route("/login")
def login():
    return render_template('login.html')  
    
@app.route("/index", methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/submit", methods=['POST'])
def get_output():
    if request.method == 'POST':
        audio_file = request.files['wavfile']
        
        if audio_file.filename.split('.')[-1] not in ['mp3', 'wav', 'ogg', 'flac', 'mpeg', 'mp4', 'weba']:
            return "Unsupported file format", 400

        # Convert the audio to WAV format
        try:
            audio = AudioSegment.from_file(audio_file)
            wav_file_path = f"static/tests/{secure_filename(audio_file.filename)}.wav"
            audio.export(wav_file_path, format="wav")
        except Exception as e:
            return f"Error converting audio: {str(e)}", 500
        
        # Perform prediction on the trimmed WAV file
        predict_result, image_path = ANN_print_prediction(wav_file_path)

    return render_template("prediction.html", prediction=predict_result, audio_path=wav_file_path, image_path=image_path)

@app.route("/chart")
def chart():
    return render_template('chart.html')     

if __name__ == '__main__':
    app.run(debug=True)
