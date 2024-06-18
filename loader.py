import librosa
import numpy as np

import os

from collections import namedtuple
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf

BASEPATH = "/home/alejandrolc/QuantumSpain/AutoQML/Data"
genres = "genres_original"
GENRES_3SEC = "genres_3sec"
IMAGES = "images_3sec_original"
AUDIO_GENRES_BASEPATH =f"{BASEPATH}/{GENRES_3SEC}"

class ImageResize(BaseEstimator, TransformerMixin):
    """
    Resizes an image
    """

    def __init__(self, size=None):
        self.size = size

    def fit(self, X, y=None):
        """returns itself"""
        if self.size == None:
            # assume image is n * width * height np array
            self.size = X.shape[1] * X.shape[2]
        return self

    def transform(self, X, y=None):
        X_resize = tf.image.resize(X[..., np.newaxis][:], (self.size, 1)).numpy()
        X_squeezed = tf.squeeze(X_resize).numpy()
        # print(X_squeezed.shape)
        return X_squeezed

def preprocess_audio():
    for genre in os.listdir(AUDIO_GENRES_BASEPATH):
        AUDIO_GENRES_PATH = f"{AUDIO_GENRES_BASEPATH}/{genre}"
        for f in os.listdir(f"{BASEPATH}/{IMAGES}/{genre}"):
            os.remove(f"{BASEPATH}/{IMAGES}/{genre}/{f}")
        for audio in os.listdir(AUDIO_GENRES_PATH):
            AUDIO_PATH = f"{AUDIO_GENRES_PATH}/{audio}"
            y, sr = librosa.load(AUDIO_PATH)
            mel = librosa.feature.melspectrogram(y=y, sr=sr)

            mel = np.array(librosa.power_to_db(mel, ref=np.max), dtype='float64')
            
            np.savetxt(f"{BASEPATH}/{IMAGES}/{genre}/{audio[:len(audio)-4]}.csv", mel, delimiter=',')

def get_spectrogram_dataset(genres=["country", "rock"]):
    x, y = [], []

    IMAGES_PATH =f"{BASEPATH}/{IMAGES}"
    for genre in genres:
        IMAGE_GENRE_PATH = f"{IMAGES_PATH}/{genre}"
        for image in os.listdir(IMAGE_GENRE_PATH):
            IMAGE_PATH = f"{IMAGE_GENRE_PATH}/{image}"

            img_array = np.array(np.loadtxt(IMAGE_PATH, dtype='float64', delimiter=','))

            x.append(img_array)
            y.append(0 if genre == genres[0] else 1)

    y = np.array(y)
    x = np.array(x)

    return x, y

## data preproccessing
def dataset(genre_pair = ["country", "rock"]):
    pipeline_image = Pipeline([
        ("scaler", ImageResize(size=256))
    ])
    
    Samples = namedtuple("samples", ["x_train", "x_test", "y_train", "y_test"])

    x, y = get_spectrogram_dataset(genre_pair)

    image_samples_raw = Samples(*train_test_split(x, y, train_size=0.7))

    image_samples_preprocessed = Samples(
        pipeline_image.fit_transform(image_samples_raw.x_train),
        pipeline_image.transform(image_samples_raw.x_test),
        image_samples_raw.y_train,
        image_samples_raw.y_test,
    )

    return image_samples_preprocessed

if __name__ == "__main__":
    # preprocess_audio()
    pass