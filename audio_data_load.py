import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import eyed3


def features_extractor(file):
    duration_s = eyed3.load(file).info.time_secs
    audio, sample_rate = librosa.load(
        file, res_type="kaiser_fast", sr=48000, mono=True, duration=duration_s - 0.5
    )
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


class AudioLoad:
    def __init__(self):
        pass

    def load_audio_data(self, csv_path):
        self.csv_path = csv_path
        metadata = pd.read_csv(self.csv_path)
        extracted_features = []
        for index_num, row in tqdm(metadata.iterrows()):
            file_name = str(row["filename"])  # name of all files
            final_class_labels = row["class"]
            data = features_extractor(file_name)
            extracted_features.append([data, final_class_labels])

        extracted_features_df = pd.DataFrame(
            extracted_features, columns=["feature", "class"]
        )

        X = np.array(extracted_features_df["feature"].tolist())
        y = np.array(extracted_features_df["class"].tolist())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        return X_train, X_test, y_train, y_test
