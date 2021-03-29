import validators
from pytube import YouTube
import pytube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import librosa
import math
import json
from tensorflow import keras
import numpy as np
import re
from urllib import parse as prs


SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
JSON_PATH = "./data/data_test.json"


def validate(link):
    # validate that it is specifically a YouTube link with a regex
    regex = re.compile(r'youtu')
    if regex.search(link):

        if validators.url(link) != True:
            return False
        else:
            print('URL valid.')
            return True
    else:
        return False

def saveAudio(link):
    try:
        # validate that the video is not longer than 15 minutes
        vid = pytube.YouTube(link)
        len = vid.length
        if len <= 900:
            # convert to mp4
            YouTube(link) \
                .streams \
                .filter(file_extension='mp4', only_audio=True) \
                .first() \
                .download(output_path='audio', filename='song')
            return True
        else:
            # video is longer than 15 minutes
            msg = 'The video is too long - please enter a track shorter than 15 minutes'
            return msg
    except:
        # if youtube link not recognised
        return False


def clipAudio():
    # extracts 30 second clip from the first minute to account for silence/long intro
    ffmpeg_extract_subclip("./audio/song.mp4", 60, 90, targetname="./audio/clip.mp4")
    return True


def saveMFCC(num_segments=5, hop_length=512, n_mfcc=13, n_fft=2048):
    # dictionary to store data
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # load track
    signal, sr = librosa.load('./audio/clip.mp4', sr=SAMPLE_RATE)

    # process segments extracting mfcc and storing data
    for s in range(num_segments):
        start_sample = num_samples_per_segment * s
        finish_sample = start_sample + num_samples_per_segment

        # store mfcc per segment if it has the expected length
        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                    sr=sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)

        mfcc = mfcc.T
        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
            data["labels"].append(s)


    with open(JSON_PATH, "w") as fp:
        json.dump(data, fp, indent=4)
    return True


def predict(model):
    # load data
    X = load_data(JSON_PATH)
    X = X[0][:130]

    # add new axes
    X = X[np.newaxis,...,np.newaxis]

    # predict
    prediction = model.predict(X)

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)
    #print("Predicted index: {}".format(predicted_index))
    return predicted_index


def load_data(dataset_path):
    with open(dataset_path, 'r')as fp:
        data = json.load(fp)

    # convert list into numpy arrays
    X = np.array(data['mfcc'])

    return X

def full_prediction(link):
    # mapping
    mapping = ['jazz', 'rock', 'disco', 'pop', 'country', 'reggae', 'hiphop', 'blues', 'metal', 'classical']

    # decode url
    link = prs.unquote(link)
    link = 'https://www.youtube.com/watch?v=' + link
    # validate url
    valid = validate(link)

    # prompt for valid link until provided
    if valid == False:
        # get youtube link
        fail = "URL Invalid. Please enter the song\'s YouTube link:"
        return fail

    # save audio, sending the error back to the calling process if the clip
    # is too long
    b = saveAudio(link)
    if b !=True:
        return b

    clipAudio()

    # preprocess audio for classification
    saveMFCC()

    # load trained model and make prediction
    model = keras.models.load_model('./classifier_model')
    prediction = int(predict(model))
    pred = mapping[prediction]
    return pred


