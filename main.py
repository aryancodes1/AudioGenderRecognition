from keras.models import load_model
import librosa
import numpy as np
from playsound import playsound
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv


def features_extractor(file):
    audio, sample_rate = librosa.load(
        file, res_type="kaiser_fast", sr=48000, mono=True, duration=1.7
    )
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


class AudioClassifier:
    def __init__(self) -> None:
        pass

    def find_voice_gender(self, rec_path):
        self.rec_path = rec_path
        model = load_model('/Users/arunkaul/Desktop/AudioSignal/AudioGenderRecognition/model.keras')#Add Model.keras Path
        model.compile(
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            optimizer="adam",
        )
        prediction_feature = features_extractor(self.rec_path)
        prediction_feature = prediction_feature.reshape(1, -1)
        p = model.predict(prediction_feature)[0]
        playsound(self.rec_path)
        if p[0] > 0.8:
            return "Your Voice Sounds Masculine"
        elif p[1] > 0.8:
            return "Your Voice Sounds Feminine"

    def record_voice(self, filename):
        self.filename = filename
        recording = sd.rec(int(5 * 48000), samplerate=48000, channels=1)
        print("Start Speaking")
        sd.wait()
        print("End")
        write(self.filename, 48000, recording)
