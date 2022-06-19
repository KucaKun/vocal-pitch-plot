from multiprocessing import Pool
import argparse
from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
from files import loadFromFile, saveToFile
from notes import (
    NOTE_NAMES,
    NOTE_VALUES,
    NOTE_VALUES_FLOAT,
    color_map,
    v_get_note_delta,
    v_freq_to_note_float,
)

SAMPLE_RATE = 96000
RESOLUTION_HZ = 5
WINDOW_SIZE = SAMPLE_RATE // RESOLUTION_HZ
MAX_NOTE = max(NOTE_VALUES)
MIN_NOTE = min(NOTE_VALUES)
FREQUENCIES = np.fft.fftfreq(WINDOW_SIZE, 1 / SAMPLE_RATE)[: WINDOW_SIZE // 2]
resolution = SAMPLE_RATE // WINDOW_SIZE


def harmonic_product_spectrum(chunk):
    num_prod = 3
    mag = np.abs(np.fft.fft(chunk).real)[: WINDOW_SIZE // 2]
    smallestLength = int(np.ceil(len(mag) / num_prod))
    y = mag[:smallestLength].copy()
    for i in range(2, num_prod + 1):
        y *= mag[::i][:smallestLength]

    return FREQUENCIES[y.argmax()]


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--files", help="input files", nargs="+", default=".\\test.wav"
    )
    parser.add_argument(
        "-l", "--live", help="play animation", action="store_true", default=False
    )
    args = parser.parse_args()
    samples_from_files = []
    for f in args.files:
        samples_from_files.append(np.array(loadFromFile(f)).flatten())

    p = pyaudio.PyAudio()
    stream = p.open(format=1, channels=1, rate=SAMPLE_RATE, output=True)

    fig, axes = plt.subplots(len(samples_from_files), 1, figsize=(16, 12))
    ctr = 0
    for samples in samples_from_files:
        chunked_samples = np.array_split(samples, len(samples) // WINDOW_SIZE)
        freqs = np.array(
            Pool(processes=14).map(harmonic_product_spectrum, chunked_samples),
            dtype=np.ndarray,
        )
        print(freqs[-10:-1])
        mapped_freqs = v_freq_to_note_float(freqs).astype(np.float64)
        mapped_freqs = np.clip(
            mapped_freqs, min(NOTE_VALUES_FLOAT), max(NOTE_VALUES_FLOAT)
        )
        axes[ctr].clear()
        axes[ctr].set_title(args.files[ctr])
        axes[ctr].set_xlim(0, len(freqs))
        axes[ctr].set_ylim(min(mapped_freqs), max(mapped_freqs))
        axes[ctr].set_yticks(NOTE_VALUES_FLOAT, NOTE_NAMES)
        axes[ctr].scatter(
            np.arange(len(freqs)),
            mapped_freqs,
            c=color_map(v_get_note_delta(mapped_freqs).flatten()),
            s=len(freqs) // RESOLUTION_HZ,
            marker="_",
        )
        axes[ctr].grid(alpha=0.1)
        ctr += 1
    if args.live:
        lines = [
            axes[x].axvline(x=0, color="r") for x in range(len(samples_from_files))
        ]
        chunked_samples = np.sum(
            [
                np.array_split(
                    samples_from_files[x] / len(samples_from_files),
                    len(samples_from_files[x]) // WINDOW_SIZE,
                )
                for x in range(len(samples_from_files))
            ],
            axis=0,
        )

        def update(i):
            if i < len(chunked_samples):
                chunk = chunked_samples[i] / 10
                stream.write(chunk.tobytes())
                for line in lines:
                    line.set_xdata(i)
            return lines

        ani = animation.FuncAnimation(
            fig,
            update,
            interval=RESOLUTION_HZ / 1000,
            blit=True,
            repeat=True,
            frames=len(chunked_samples),
        )
    plt.show()
