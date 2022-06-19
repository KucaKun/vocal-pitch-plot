import os
from scipy.io import wavfile


def getLastGenerated():
    numbers = []
    for filename in os.listdir(".\\waves\\"):
        if "generated" in filename:
            numbers.append(int(filename.split("_")[0]))
    return max(numbers)


def saveToFile(
    frames, samp_width=3, channels=1, samplerate=96000, filename="generated.wav"
):
    # frames = am.floatsToBytes(frames)
    path = ".\\tests\\" + str(getLastGenerated() + 1) + "_" + filename
    print("**Saving file to:", path)
    wavfile.write(path, samplerate, frames)
    return


def loadFromFile(filename):
    frames = []
    print("**Reading", filename)
    samplerate, data = wavfile.read(filename)
    return data
