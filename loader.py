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
FULL_GENRES_3SEC = "full_genres_3sec"
FULL_IMAGES_3SEC = "full_images_3sec"
IMAGES = "images_3sec_original"
AUDIO_GENRES_BASEPATH =f"{BASEPATH}/{GENRES_3SEC}"
FULL_AUDIO_GENRES_BASEPATH =f"{BASEPATH}/{FULL_IMAGES_3SEC}"



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

def divide_audio():
    for genre in os.listdir(AUDIO_GENRES_BASEPATH):
        AUDIO_GENRES_PATH = f"{AUDIO_GENRES_BASEPATH}/{genre}"

        FULL_AUDIO_GENRE_BASEPATH = f"{FULL_AUDIO_GENRES_BASEPATH}/{genre}"

        if not os.path.isdir(FULL_AUDIO_GENRE_BASEPATH):
            os.makedirs(FULL_AUDIO_GENRE_BASEPATH)

        for f in os.listdir(FULL_AUDIO_GENRE_BASEPATH):
            os.remove(f"{FULL_AUDIO_GENRE_BASEPATH}/{f}")

        for audio in os.listdir(AUDIO_GENRES_PATH):
            AUDIO_PATH = f"{AUDIO_GENRES_PATH}/{audio}"
            y, sr = librosa.load(AUDIO_PATH)

            # print(y.shape)
            shift = y.shape[0]//10
        
            for i in range(10):
                mel = librosa.feature.melspectrogram(y=y[i*shift:(i+1)*shift], sr=sr)

                mel = np.array(librosa.power_to_db(mel, ref=np.max), dtype='float64')
                
                np.savetxt(f"{FULL_AUDIO_GENRE_BASEPATH}/{audio[:len(audio)-4]}{i}.csv", mel, delimiter=',')

def preprocess_yaseen():
    BASEPATH = "/home/alejandrolc/QuantumSpain/AutoQML/Data/Yaseen"
    ORIGINAL = "original"
    IMAGES = "images"
    AUDIO_BASEPATH =f"{BASEPATH}/{ORIGINAL}"

    for audio_type in os.listdir(AUDIO_BASEPATH):
        AUDIO_GENRES_PATH = f"{AUDIO_BASEPATH}/{audio_type}"
        for f in os.listdir(f"{BASEPATH}/{IMAGES}/{audio_type}"):
            os.remove(f"{BASEPATH}/{IMAGES}/{audio_type}/{f}")
        for audio in os.listdir(AUDIO_GENRES_PATH):
            AUDIO_PATH = f"{AUDIO_GENRES_PATH}/{audio}"
            try:
                y, sr = librosa.load(AUDIO_PATH)
                # print(librosa.get_duration(y=y, sr=sr), len(y))
                if len(y) > 3 * sr:
                    y = y[:3*sr]
                else:
                    # print(len(y))
                    y_ = np.zeros((3*sr))
                    y_[:len(y)] = y
                    y = y_

                print(len(y))

                mel = librosa.feature.melspectrogram(y=y, sr=sr)

                mel = np.array(librosa.power_to_db(mel, ref=np.max), dtype='float64')
                
                np.savetxt(f"{BASEPATH}/{IMAGES}/{audio_type}/{audio[:len(audio)-4]}.csv", mel, delimiter=',')
            except Exception as e:
                with open("errors.txt", mode='a') as f:
                    f.write(f"{audio}\n {e}\n")

def get_spectrogram_dataset(genres=["country", "rock"]):
    x, y = [], []

    IMAGES_PATH =f"{BASEPATH}/{FULL_IMAGES_3SEC}"   # FULL_IMAGES_3SEC
    for genre in genres:
        IMAGE_GENRE_PATH = f"{IMAGES_PATH}/{genre}"
        for image in os.listdir(IMAGE_GENRE_PATH):
            IMAGE_PATH = f"{IMAGE_GENRE_PATH}/{image}"

            img_array = np.array(np.loadtxt(IMAGE_PATH, dtype='float64', delimiter=','))
            
            if np.any(img_array):
                x.append(img_array)
                y.append(0 if genre == genres[0] else 1)

    y = np.array(y)
    x = np.array(x)

    return x, y

def get_spectrogram_dataset_yaseen(types):
    x, y = [], []

    BASEPATH = "/home/alejandrolc/QuantumSpain/AutoQML/Data/Yaseen_binary"
    ORIGINAL = "original"
    IMAGES = "images"
    # AUDIO_BASEPATH =f"{BASEPATH}/{ORIGINAL}"

    # max_dur = 0

    i = 0

    IMAGES_PATH =f"{BASEPATH}/{IMAGES}"
    for genre in types:
        IMAGE_GENRE_PATH = f"{IMAGES_PATH}/{genre}"
        for image in os.listdir(IMAGE_GENRE_PATH):
            IMAGE_PATH = f"{IMAGE_GENRE_PATH}/{image}"

            img_array = np.array(np.loadtxt(IMAGE_PATH, dtype='float64', delimiter=','))

            # max_dur = np.max(img_array.shape[1], max_dur)
            # print(img_array.shape)
            # if genre == "N":
            #     x.append(img_array)
            #     x.append(img_array)
            #     x.append(img_array)
            #     y.append(0 if genre == types[0] else 1)
            #     y.append(0 if genre == types[0] else 1)
            #     y.append(0 if genre == types[0] else 1)

            if i % 4 == 0 and genre == "NOTN":
                x.append(img_array)
                y.append(0 if genre == types[0] else 1)

            if genre == "N":
                x.append(img_array)
                y.append(0 if genre == types[0] else 1)

            i += 1


    y = np.array(y)
    x = np.array(x)

    print(x.shape)

    # lens = [len(l) for l in x]
    # print(lens)
    # maxlen = max(lens)
    # print(maxlen)
    # x_ = -80*np.ones((len(x), maxlen))
    # mask = np.arange(maxlen) < np.array(lens)[:,None]
    # x_[mask] = np.concatenate(x)

    # x = x_

    return x, y

def dataset_yaseen(pair):
    pipeline_image = Pipeline([
        ("scaler", ImageResize(size=256))
    ])
    
    Samples = namedtuple("samples", ["x_train", "x_test", "y_train", "y_test"])

    x, y = get_spectrogram_dataset_yaseen(pair)

    image_samples_raw = Samples(*train_test_split(x, y, train_size=0.7))

    image_samples_preprocessed = Samples(
        pipeline_image.fit_transform(image_samples_raw.x_train),
        pipeline_image.transform(image_samples_raw.x_test),
        image_samples_raw.y_train,
        image_samples_raw.y_test,
    )

    return image_samples_preprocessed

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

def full_dataset(genre_pair):
    pipeline_image = Pipeline([
        ("scaler", ImageResize(size=256))
    ])
    
    Samples = namedtuple("samples", ["x", "y"])

    x, y = get_spectrogram_dataset(genre_pair)

    image_samples_raw = Samples(x, y)

    image_samples_preprocessed = Samples(
        pipeline_image.fit_transform(image_samples_raw.x),
        image_samples_raw.y
    )

    return image_samples_preprocessed

if __name__ == "__main__":
    # preprocess_yaseen()
    divide_audio()
    # preprocess_audio()
    pass