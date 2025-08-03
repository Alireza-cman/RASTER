import numpy as np
from ssqueezepy import ssq_cwt, ssq_stft
from ssqueezepy.experimental import scale_to_freq
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import norm
from scipy.optimize import curve_fit
from ssqueezepy import ssq_cwt, ssq_stft
from utils import RASTER
from utils import classifier as CLF
from scipy.stats import entropy
from scipy.optimize import minimize
import warnings
import matplotlib.patches as patches
import matplotlib


def viz(x, Tx, Wx):

    plt.figure(figsize=(20, 9))
    plt.subplot(2, 2, 1)
    plt.plot(x)
    plt.subplot(2, 2, 2)
    plt.plot(np.log10(np.abs(np.fft.rfft(x))))

    plt.subplot(2, 2, 3)

    plt.imshow(np.abs(Wx), aspect="auto", cmap="turbo")
    plt.subplot(2, 2, 4)
    plt.imshow(np.abs(Tx), aspect="auto", vmin=0, vmax=0.01, cmap="turbo")


def average_spectogram(x_train, verbose=True):
    n_sample, length = x_train.shape
    ـ, Sx, *_ = ssq_stft(x_train[0], fs=1)
    averaged_Tsx_result = np.zeros((n_sample, *Sx.shape))
    for i in range(n_sample):
        _, Sx, *_ = ssq_stft(x_train[i], fs=1)
        averaged_Tsx_result[i] = np.abs(Sx)
    if verbose:
        plt.imshow(
            np.abs(np.sum(averaged_Tsx_result, axis=0)), aspect="auto", cmap="turbo"
        )
    return averaged_Tsx_result


def is_evenly_distributed(averaged_Tsx_result, verbose=False, Dataset="Hello"):

    matplotlib.rcParams["pdf.fonttype"] = 42  # Make fonts editable in Illustrator
    matplotlib.rcParams["ps.fonttype"] = 42

    if averaged_Tsx_result.shape[1] < 100:
        return (True, True), {}
    data = np.abs(np.sum(averaged_Tsx_result, axis=0))
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))

    a = np.std(normalized_data, axis=1) / np.mean(normalized_data, axis=1)
    std_log = np.log10(np.std(normalized_data, axis=1))
    mean_log = np.log10(np.mean(normalized_data, axis=1))

    # New Section
    time_mean_graph = np.mean(normalized_data, axis=0)
    mean_value = np.mean(time_mean_graph)
    ppv = np.sum(time_mean_graph > mean_value) / len(time_mean_graph)

    if verbose:
        fontsize = 30
        plt.figure(figsize=(16, 8))
        # First subplot: Averaged Normalized STFT
        plt.subplot(1, 2, 1)
        plt.imshow(normalized_data, aspect="auto", cmap="turbo")
        plt.xlabel("Time", fontsize=fontsize)
        plt.ylabel("Frequency Bin", fontsize=fontsize)
        plt.title("Averaged Normalized STFT", fontsize=fontsize)
        # cbar = plt.colorbar()
        # cbar.set_label('Normalized Amplitude', fontsize=14)

        # INSERT_YOUR_CODE
        # Set x-ticks at positions divisible by 100, up to the max time index
        time_len = normalized_data.shape[1]
        xticks = [i for i in range(0, time_len, 400)]
        plt.xticks(xticks, fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        # Second subplot: Distribution statistics
        plt.subplot(1, 2, 2)
        freq_axis = np.arange(normalized_data.shape[0])
        plt.plot(
            freq_axis,
            np.log10(np.mean(normalized_data, axis=1)),
            label="Mean",
            alpha=0.3,
            linewidth=1,
            color="blue",
        )
        plt.plot(
            freq_axis,
            np.log10(np.std(normalized_data, axis=1)),
            label="Std",
            alpha=0.3,
            linewidth=1,
            color="green",
        )
        plt.plot(
            freq_axis,
            std_log - mean_log,
            label="Std - Mean",
            linewidth=3,
            color="black",
        )
        # Draw a horizontal threshold at -1 to indicate ISD algorithm pass/fail
        plt.axhline(
            y=-1, color="red", linestyle="--", linewidth=2, label="Threshold (-1)"
        )

        plt.xlabel("Frequency Bin", fontsize=fontsize)
        plt.title("Log-Scale Mean & Std Across Frequency", fontsize=fontsize)
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        # plt.legend(fontsize=16, loc='best', title_fontsize=16)
        # plt.legend(fontsize=13, loc='upper left', title='Statistics', title_fontsize=14, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        # INSERT_YOUR_CODE
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))

        plt.tight_layout()
        plt.savefig(Dataset + ".pdf", bbox_inches="tight", dpi=300)

    result = std_log - mean_log
    a = result < -1
    what_to_return = np.sum(a) > 0
    if ppv > 0.6:
        what_to_return2 = True
    else:
        what_to_return2 = False

    ultimate_return = False
    if what_to_return2 == True:
        ultimate_return = True
    else:
        if what_to_return == True:
            ultimate_return = True
        else:
            ultimate_return = False
    return ultimate_return


########## Second Approach ##########


# Define a Gaussian function
def gaussian(params, x):
    a, b, c = params
    return a * np.exp(-((x - b) ** 2) / (2 * c**2))


# Define residuals for least squares
def residuals(params, x, y):
    return gaussian(params, x) - y
