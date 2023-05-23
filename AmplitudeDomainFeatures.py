import librosa
import numpy as np
import matplotlib.pyplot as plt
import os


class TimeDomainFeatures:
    def __init__(self, audio_paths):
        self.audio_paths = audio_paths
        self.audio_data = []
        self.sr = []
        self.duration = []
        self.time = []

        for audio_path in self.audio_paths:
            audio, sr = librosa.load(audio_path, sr=None)
            self.audio_data.append(audio)
            self.sr.append(sr)
            self.duration.append(len(audio) / sr)
            self.time.append(np.linspace(0, self.duration[-1], len(audio)))

    def plot_amplitude_time(self):
        plt.figure(figsize=(12, 4))

        for i, audio in enumerate(self.audio_data):
            plt.plot(self.time[i], audio, label=f'Audio {i+1}')

        plt.xlabel('Waktu (detik)')
        plt.ylabel('Amplitudo')
        plt.title('Representasi Amplitudo-Waktu')
        plt.legend()
        plt.show()

    def compute_energy(self):
        energies = []

        for audio in self.audio_data:
            energy = np.square(audio)
            energies.append(energy)

        return energies

    def plot_energy(self):
        energies = self.compute_energy()

        plt.figure(figsize=(12, 4))

        for i, energy in enumerate(energies):
            plt.plot(self.time[i], energy, label=f'Audio {i+1}')

        plt.xlabel('Waktu (detik)')
        plt.ylabel('Energi')
        plt.title('Representasi Energi')
        plt.legend()
        plt.show()

    def compute_zero_crossing_rate(self):
        zero_crossing_rates = []

        for audio in self.audio_data:
            zero_crossings = librosa.zero_crossings(audio, pad=False)
            zero_crossing_rate = np.mean(zero_crossings)
            zero_crossing_rates.append(zero_crossing_rate)

        return zero_crossing_rates

    def compute_silence_ratio(self, threshold=0.01):
        silence_ratios = []

        for audio in self.audio_data:
            above_threshold = np.sum(np.abs(audio) > threshold)
            silence_ratio = 1 - (above_threshold / len(audio))
            silence_ratios.append(silence_ratio)

        return silence_ratios


# Contoh penggunaan
# Ganti dengan path ke tiga folder yang berisi file audio
folder_paths = ['datasets/happy', 'datasets/neutral', 'datasets/sad']

audio_paths = []
for folder_path in folder_paths:
    audio_files = os.listdir(folder_path)
    for audio_file in audio_files:
        audio_paths.append(os.path.join(folder_path, audio_file))

time_domain = TimeDomainFeatures(audio_paths)
time_domain.plot_amplitude_time()
time_domain.plot_energy()

zero_crossing_rates = time_domain.compute_zero_crossing_rate()
silence_ratios = time_domain.compute_silence_ratio()

for i in range(len(audio_paths)):
    print(f'Audio {i+1}')
    print(f'Zero Crossing Rate: {zero_crossing_rates[i]}')
    print(f'Silence Ratio: {silence_ratios[i]}')
    print('-------------------')
