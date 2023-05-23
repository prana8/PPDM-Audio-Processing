import AmplitudeDomainFeatures
import FrequencyDomainFeatures
import MFCCExtractor
import os

# Path ke tiga folder yang berisi file audio
folder_paths = ['datasets/happy', 'datasets/neutral', 'datasets/sad']

# Membaca file audio dari folder dan menyimpan path-nya dalam list audio_paths
audio_paths = []
for folder_path in folder_paths:
    audio_files = os.listdir(folder_path)
    for audio_file in audio_files:
        audio_paths.append(os.path.join(folder_path, audio_file))

# Menggunakan class AmplitudeDomainFeatures
amplitude_domain = AmplitudeDomainFeatures(audio_paths)
amplitude_domain.plot_amplitude_time()
energies = amplitude_domain.compute_energy()
zero_crossing_rates = amplitude_domain.compute_zero_crossing_rate()
silence_ratios = amplitude_domain.compute_silence_ratio()

# Menggunakan class FrequencyDomainFeatures
frequency_domain = FrequencyDomainFeatures(audio_paths)
frequency_domain.plot_sound_spectrum()
bandwidths = frequency_domain.compute_bandwidth()
spectral_centroids = frequency_domain.compute_spectral_centroid()

# Menggunakan class MFCCExtractor
mfcc_extractor = MFCCExtractor(audio_paths)
mfccs = mfcc_extractor.extract_mfcc()

# Menampilkan hasil
for i in range(len(audio_paths)):
    print(f'Audio {i+1}')
    print(f'Amplitude Domain - Energy: {energies[i]}')
    print(f'Amplitude Domain - Zero Crossing Rate: {zero_crossing_rates[i]}')
    print(f'Amplitude Domain - Silence Ratio: {silence_ratios[i]}')
    print(f'Frequency Domain - Bandwidth: {bandwidths[i]}')
    print(f'Frequency Domain - Spectral Centroid: {spectral_centroids[i]}')
    print(f'MFCCs: {mfccs[i]}')
    print('-------------------')
