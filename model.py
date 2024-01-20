from audio_data_load import AudioLoad
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D
from tensorflow.keras.optimizers import Adam
import librosa
import numpy as np
import eyed3

ad = AudioLoad()
X_data, Y_data, X_label, Y_label = ad.load_audio_data(csv_path='/Users/arunkaul/Desktop/AudioSignal/AudioGenderRecognition/cv-other-test .csv')


print(X_data.shape)

model = Sequential()

model.add(Dense(2048, input_shape=(40,), activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(256, activation="relu", kernel_regularizer="l2"))
model.add(Dense(2, activation="softmax"))
model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    optimizer=tf.keras.optimizers.Adam(0.007),
)
model.fit(X_data, X_label, validation_data=(Y_data, Y_label), epochs=75)


def features_extractor(file):
    duration_s = eyed3.load(file).info.time_secs
    audio, sample_rate = librosa.load(
        file, res_type="kaiser_fast", sr=48000, mono=True, duration=duration_s - 0.5
    )
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

model.save("model1.keras")
