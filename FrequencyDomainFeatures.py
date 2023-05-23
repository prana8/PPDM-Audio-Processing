import librosa
import numpy as np
import matplotlib.pyplot as plt
import os


class FrequencyDomainFeatures:
    def __init__(self, audio_paths):
        self.audio_paths = audio_paths
        self.audio_data = []
        self.sr = []
        self.duration = []
        self.fft_size = 2048

        for audio_path in self.audio_paths:
            audio, sr = librosa.load(audio_path, sr=None)
            self.audio_data.append(audio)
            self.sr.append(sr)
            self.duration.append(len(audio) / sr)

    def plot_sound_spectrum(self):
        plt.figure(figsize=(12, 6))

        for i, audio in enumerate(self.audio_data):
            spectrum = np.abs(librosa.stft(audio, n_fft=self.fft_size))
            log_spectrum = librosa.amplitude_to_db(spectrum, ref=np.max)

            plt.subplot(len(self.audio_data), 1, i+1)
            librosa.display.specshow(
                log_spectrum, sr=self.sr[i], hop_length=self.fft_size//4, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Representasi Spectrum Suara - Audio {i+1}')

        plt.tight_layout()
        plt.show()

    def compute_bandwidth(self):
        bandwidths = []

        for i, audio in enumerate(self.audio_data):
            spectrum = np.abs(librosa.stft(audio, n_fft=self.fft_size))
            frequencies = librosa.fft_frequencies(
                sr=self.sr[i], n_fft=self.fft_size)
            amplitude_spectrum = np.sum(spectrum, axis=1)
            weighted_mean = np.sum(
                frequencies * amplitude_spectrum) / np.sum(amplitude_spectrum)
            squared_differences = np.square(frequencies - weighted_mean)
            bandwidth = np.sum(squared_differences *
                               amplitude_spectrum) / np.sum(amplitude_spectrum)
            bandwidths.append(bandwidth)

        return bandwidths

    def compute_spectral_centroid(self):
        spectral_centroids = []

        for i, audio in enumerate(self.audio_data):
            spectrum = np.abs(librosa.stft(audio, n_fft=self.fft_size))
            frequencies = librosa.fft_frequencies(
                sr=self.sr[i], n_fft=self.fft_size)
            amplitude_spectrum = np.sum(spectrum, axis=1)
            spectral_centroid = np.sum(
                frequencies * amplitude_spectrum) / np.sum(amplitude_spectrum)
            spectral_centroids.append(spectral_centroid)

        return spectral_centroids


# Contoh penggunaan
folder_paths = ['datasets/happy', 'datasets/neutral', 'datasets/sad']
audio_paths = []
for folder_path in folder_paths:
    audio_files = os.listdir(folder_path)
    for audio_file in audio_files:
        audio_paths.append(os.path.join(folder_path, audio_file))

frequency_domain = FrequencyDomainFeatures(audio_paths)
frequency_domain.plot_sound_spectrum()

bandwidths = frequency_domain.compute_bandwidth()
spectral_centroids = frequency_domain.compute_spectral_centroid()

for i in range(len(audio_paths)):
    print(f'Audio {i+1}')
    print(f'Bandwidth: {bandwidths[i]}')
    print(f'Spectral Centroid: {spectral_centroids[i]}')
    print('-------------------')
