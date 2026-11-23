"""Reduce vocal, drum and bass bleed in a separated guitar stem.

The six-stem Demucs guitar estimate is the weakest stem and tends to carry
vocals, drums and bass once they enter. The other stems are independent
estimates of that same material, so bins where they clearly dominate are
attenuated with a soft spectral mask. The raw stem is never modified.
"""
import numpy as np

N_FFT = 4096
HOP = 1024
INTERFERENCE_WEIGHT = 1.0
MASK_FLOOR = .08
MASK_POWER = 1.0


def _stft(channel):
    import librosa
    return librosa.stft(channel, n_fft=N_FFT, hop_length=HOP, window="hann")


def suppress_bleed(guitar, interferers, weight=INTERFERENCE_WEIGHT, floor=MASK_FLOOR, power=MASK_POWER):
    """Return guitar (channels, samples) with bins dominated by interferers attenuated.

    `interferers` is a list of arrays with the same shape as `guitar`. Time
    alignment and length are preserved; no mixing between channels occurs.
    """
    import librosa
    guitar = np.asarray(guitar, dtype=np.float32)
    output = np.empty_like(guitar)
    for channel in range(guitar.shape[0]):
        spectrum = _stft(guitar[channel])
        guitar_power = np.abs(spectrum) ** 2
        interference = np.zeros_like(guitar_power)
        for other in interferers:
            interference += np.abs(_stft(np.asarray(other[channel], dtype=np.float32))) ** 2
        ratio = guitar_power / (guitar_power + weight * interference + 1e-12)
        mask = floor + (1 - floor) * ratio ** power
        output[channel] = librosa.istft(spectrum * mask, hop_length=HOP, window="hann", length=guitar.shape[1])
    return output
