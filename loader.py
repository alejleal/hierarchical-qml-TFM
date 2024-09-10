import librosa
import numpy as np
from scipy.fftpack import dct

import os

from collections import namedtuple
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf

from enum import Enum

class DatasetName(Enum):
    GTZAN = 0
    YASEEN = 1

DATAPATH = "/home/alejandrolc/QuantumSpain/AutoQML/Dataset"
GTZAN_PATH = f"{DATAPATH}/GTZAN"
ORIG_GENRES_PATH = f"{GTZAN_PATH}/Data/genres_original"
GTZAN_PR_PATH = f"{GTZAN_PATH}/PreprocessedData"

PP_PATH = lambda ds: f"{DATAPATH}/{ds}/PreprocessedData"
RD_PATH = lambda ds: f"{DATAPATH}/{ds}/Data"


# genres = "genres_original"
GENRES_3SEC = "genres_3sec"
FULL_GENRES_3SEC = "full_genres_3sec"
FULL_IMAGES_3SEC = "full_images_3sec"
IMAGES = "images_3sec_original"
GENRES3_PATH =f"{DATAPATH}/{GENRES_3SEC}"
FULL_AUDIO_GENRES_BASEPATH =f"{DATAPATH}/{FULL_IMAGES_3SEC}"


"""
images_original -> espectrogramas de mel como imagen (original ds)
images_3sec_original -> .csv con los 3 primeros segundos (original ds)
images_3sec -> imagenes de los 3 primeros segundos, como imagen y reducidas a 8x32

genres_original -> audios .wav con los audios completos (original ds)
genres_3sec ->  audios .wav de los 3 primeros segundos de cada audio (original ds)

full_images_3sec -> .csv con los espectrogramas de los audios divididos en partes de 3 segundos sin solapamiento (10 audios de 3 segundos por cada audio)


Rediseño de la estructura de los datos:
./Datasets
    /GTZAN
        /Original
            /images_original
            /images_3sec_original
            /genres_original
            /genres_3sec
        /Data
        /PreprocessedData
            /mel_x -> csv de los espectrogramas de cada audio extrayendo x partes de 3 segundos
            /mfcc_x -> csv de los mfccs de cada audio extrayendo x partes de 3 segundos
    /Yaseen
        /Data
        /PreprocessedData
"""



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
    
class ImageRepetition(BaseEstimator, TransformerMixin):
    """
    Resizes an image
    """

    def __init__(self, size=None, times=1):
        self.size = size
        self.times = times

    def fit(self, X, y=None):
        """returns itself"""
        if self.size == None:
            # assume image is n * width * height np array
            self.size = X.shape[0] * X.shape[1]
        return self

    def transform(self, X, y=None):
        return np.tile(X, self.times)
    

def preprocess_audio_gen(ds, method, n):
    """
    ds: dataset (gtzan, yaseen)
    method: metodo para extraer las features (mel, mfccs, ...)
        mel  -> librosa.feature.melspectrogram(y=y, sr=sr)
                np.array(librosa.power_to_db(mel, ref=np.max), dtype='float32')
        mfcc -> librosa.feature.mfcc(y=y, sr=sr)
    n: numero de subaudios que se extraen de cada audio
    """

    pp_datapath = f"{PP_PATH(ds)}/{method}_{n}"
    audio_path = RD_PATH(ds)

    for group in os.listdir(audio_path):
        csv_group_path = f"{pp_datapath}/{group}"
        audio_group_path = f"{audio_path}/{group}"

        if not os.path.isdir(csv_group_path):
            os.makedirs(csv_group_path)
        else:
            # Borra todo lo que se encuentre en el directorio con los datos procesados
            for f in os.listdir(csv_group_path):
                os.remove(f"{csv_group_path}/{f}")

        for wav in os.listdir(audio_group_path):
            wav_path = f"{audio_group_path}/{wav}"

            try:
                y, sr = librosa.load(wav_path)
            except Exception:
                continue

            # TODO: Hay cambios importantes entre GTZAN y Yaseen. Revisar detenidamente
            # sec2ts = 30/y.shape[0]
            audio_len = sr*3
            shift = int(min(3, 27/(n-1) if n > 1 else 0)*sr)

            # shift = y.shape[0]//10
            for i in range(n):
                # TODO: Comprobar el tipo de metodo que es y adaptarlo acorde
                if method == 'mel':
                    spec = librosa.feature.melspectrogram(y=y[i*shift:i*shift+audio_len], sr=sr)

                    # mel = librosa.feature.melspectrogram(y=y, sr=sr)
                    spec = np.array(librosa.power_to_db(spec, ref=np.max), dtype='float32')
                    np.savetxt(f"{csv_group_path}/{wav[:len(wav)-4]}{i}.csv", spec, delimiter=',')

                elif method == 'dct':
                    spec = dct(y[i*shift:i*shift+audio_len])
                    np.savetxt(f"{csv_group_path}/{wav[:len(wav)-4]}{i}.csv", spec, delimiter=',')

                elif method == 'mfcc':
                    # spec = librosa.feature.melspectrogram(y=y[i*shift:i*shift+audio_len], sr=sr)

                    # # mel = librosa.feature.melspectrogram(y=y, sr=sr)
                    # spec = librosa.power_to_db(spec, ref=np.max)
                    spec = librosa.feature.mfcc(y=y[i*shift:i*shift+audio_len], sr=sr)
                    np.savetxt(f"{csv_group_path}/{wav[:len(wav)-4]}{i}.csv", spec, delimiter=',')
                    # print(y[i*shift:i*shift+audio_len].shape, mel.shape)

def preprocess_audio():
    for genre in os.listdir(GENRES3_PATH):
        GENRES_PATH = f"{GENRES3_PATH}/{genre}"

        # Borrar todo lo que se encuentre en el directorio con los datos procesados
        for f in os.listdir(f"{DATAPATH}/{IMAGES}/{genre}"):
            os.remove(f"{DATAPATH}/{IMAGES}/{genre}/{f}")

        for audio in os.listdir(GENRES_PATH):
            AUDIO_PATH = f"{GENRES_PATH}/{audio}"
            y, sr = librosa.load(AUDIO_PATH)
            mel = librosa.feature.melspectrogram(y=y, sr=sr)

            mel = np.array(librosa.power_to_db(mel, ref=np.max), dtype='float32')
            
            np.savetxt(f"{DATAPATH}/{IMAGES}/{genre}/{audio[:len(audio)-4]}.csv", mel, delimiter=',')

def divide_audio():
    for genre in os.listdir(GENRES3_PATH):
        GENRE_PATH = f"{GENRES3_PATH}/{genre}"

        FULL_AUDIO_GENRE_BASEPATH = f"{FULL_AUDIO_GENRES_BASEPATH}/{genre}"

        if not os.path.isdir(FULL_AUDIO_GENRE_BASEPATH):
            os.makedirs(FULL_AUDIO_GENRE_BASEPATH)

        for f in os.listdir(FULL_AUDIO_GENRE_BASEPATH):
            os.remove(f"{FULL_AUDIO_GENRE_BASEPATH}/{f}")

        for audio in os.listdir(GENRE_PATH):
            AUDIO_PATH = f"{GENRE_PATH}/{audio}"
            y, sr = librosa.load(AUDIO_PATH)

            # print(y.shape)
            shift = y.shape[0]//10
        
            for i in range(10):
                mel = librosa.feature.melspectrogram(y=y[i*shift:(i+1)*shift], sr=sr)

                mel = np.array(librosa.power_to_db(mel, ref=np.max), dtype='float32')
                
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

                mel = np.array(librosa.power_to_db(mel, ref=np.max), dtype='float32')
                
                np.savetxt(f"{BASEPATH}/{IMAGES}/{audio_type}/{audio[:len(audio)-4]}.csv", mel, delimiter=',')
            except Exception as e:
                with open("errors.txt", mode='a') as f:
                    f.write(f"{audio}\n {e}\n")

def get_spectrogram_dataset(ds, genres, method, n=10):
    x, y = [], []

    # IMAGES_PATH = f"{DATAPATH}/{IMAGES}" if not extended else f"{DATAPATH}/{FULL_IMAGES_3SEC}"

    pp_datapath = f"{PP_PATH(ds)}/{method}_{n}"

    shapes = []

    for genre in genres:
        csv_genre_path = f"{pp_datapath}/{genre}"
        for image in os.listdir(csv_genre_path):
            csv_path = f"{csv_genre_path}/{image}"

            img_array = np.array(np.loadtxt(csv_path, dtype='float32', delimiter=','))
            # print(img_array.shape)

            if method == 'mel' or method == 'mfcc':
                shapes.append(img_array.shape[1])
            elif method == 'dct':
                shapes.append(img_array.shape[0])
            
            if np.any(img_array):
                x.append(img_array)
                y.append(0 if genre == genres[0] else 1)

    min_shape = np.min(shapes)
    if method == 'mel' or method == 'mfcc':
        y = np.array(y)
        x = np.array([e[:, :min_shape] for e in x])
        # print(x.shape)
    else:
        min_shape = np.min(shapes)
        y = np.array(y)
        x = np.array([e[:min_shape] for e in x])[..., np.newaxis]
        # print(x.shape)

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

            img_array = np.array(np.loadtxt(IMAGE_PATH, dtype='float32', delimiter=','))

            if i % 4 == 0 and genre == "NOTN":
                x.append(img_array)
                y.append(0 if genre == types[0] else 1)

            if genre == "N":
                x.append(img_array)
                y.append(0 if genre == types[0] else 1)

            i += 1

    y = np.array(y)
    x = np.array(x)

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
def dataset(ds, pair, nfeat=256, n=1, times=1, method='mel', train_size=0.7):
    pipeline_image = Pipeline([
        ("scaler", ImageResize(size=nfeat)),
        ("repeater", ImageRepetition(times=times))
    ])
    
    Samples = namedtuple("samples", ["x_train", "x_test", "y_train", "y_test"])

    x, y = get_spectrogram_dataset(ds, pair, method, n)

    image_samples_raw = Samples(*train_test_split(x, y, train_size=train_size))

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
    # divide_audio()
    # preprocess_audio()

    # preprocess_audio_gen(DatasetName.GTZAN.name, 'mel', 1)
    # print('mel 1 finished')
    # preprocess_audio_gen(DatasetName.GTZAN.name, 'mel', 10)
    # preprocess_audio_gen(DatasetName.GTZAN.name, 'mel', 20)


    # preprocess_audio_gen(DatasetName.GTZAN.name, 'dct', 1)
    # print('dct 1 finished')
    # preprocess_audio_gen(DatasetName.GTZAN.name, 'dct', 10)

    # preprocess_audio_gen(DatasetName.GTZAN.name, 'mfcc', 1)

    # ds = dataset('GTZAN', ['rock'])

    pass