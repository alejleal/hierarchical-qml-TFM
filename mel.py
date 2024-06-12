import librosa
import numpy as np
import matplotlib.pyplot as plt

import os

BASEPATH = "/home/alejandrolc/QuantumSpain/AutoQML/Data"
GENRES = "genres_original"
GENRES_3SEC = "genres_3sec"
IMAGES = "images_3sec_original"
AUDIO_GENRES_BASEPATH =f"{BASEPATH}/{GENRES_3SEC}"

def preprocess_audio():
    # TODO: Copiar parte de lo que hay en el main

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
    # Obtener los audios del dataset
    # Quedarme con los 3 primeros segundos
    #   No hace falta si ya los tengo cortados de primera mano
    # Realizar el espectrograma de mel
        # y, sr = librosa.load(AUDIOPATH)
        # mel = librosa.feature.melspectrogram(y=y, sr=sr)
        # mel.shape # Deberia ayudarme a redimensionar
    # Realizar el pooling (max, min, avg, ..., lo que se me ocurra)
        # https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy puede ser una primera idea
    # Normalizar?
    # Dividir todo en los distintos conjuntos de entrenamiento/test/validacion
        # x_tr, x_test, y_tr, y_test = train_test_split(x, y, train_size=0.8)
        # x_val, x_test, y_val, y_test = train_test_split(x, y, train_size=0.5)

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

if __name__ == "__main__":
    # song = AudioSegment.from_wav(BASEPATH)

    # Se supone que esto deberia funcionar
    # Cortaria los audios y los guardaria en otra carpeta, suponiendo que tienen la misma estructura
    # El unico audio con el que no funciona es jazz.00054.wav por alguna razon
    # with open("errors.txt", "a") as f:
    #     for genre in os.listdir(f"{BASEPATH}/{GENRES}"):
    #         for audio in os.listdir(f"{BASEPATH}/{GENRES}/{genre}"):
    #             try:
    #                 song = AudioSegment.from_file(f"{BASEPATH}/{GENRES}/{genre}/{audio}")
    #                 seg = 3 * 1000
    #                 song[:seg].export(f"{BASEPATH}/{GENRES_3SEC}/{genre}/{audio}", format="wav")
    #             except:
    #                 f.write(audio)

    preprocess_audio()

    # print(get_spectrogram_dataset()[0][:5])
    # ds = tfds.load('gtzan')

    # ds = ds.take(1)  # Only take a single example

    # for example in ds:  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
    #     print(list(example.keys()))
    #     image = example["image"]
    #     label = example["label"]
    #     print(image.shape, label)


    ### Ejemplo
    # AUDIOPATH = "./archive/Data/genres_original/hiphop/hiphop.00023.wav"
    # y, sr = librosa.load(AUDIOPATH)
    # mel = librosa.feature.melspectrogram(y=y, sr=sr)

    # mel = mel[:,:mel.shape[1]//10]

    # fig, ax = plt.subplots()
    # S_dB = librosa.power_to_db(mel, ref=np.max)
    # img = librosa.display.specshow(S_dB, x_axis='time',
    #                      y_axis='mel', sr=sr,
    #                      fmax=8000, ax=ax)
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')
    # ax.set(title='Mel-frequency spectrogram')

    # plt.show()
