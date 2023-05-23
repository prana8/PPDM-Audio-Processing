import librosa
import numpy as np
import os


class MFCCExtractor:
    def __init__(self, audio_paths):
        self.audio_paths = audio_paths
        self.audio_data = []
        self.sr = []

        for audio_path in self.audio_paths:
            audio, sr = librosa.load(audio_path, sr=None)
            self.audio_data.append(audio)
            self.sr.append(sr)

    def extract_mfcc(self):
        mfccs = []

        for i, audio in enumerate(self.audio_data):
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sr[i])
            mfccs.append(mfcc)

        return mfccs


# Contoh penggunaan
# Ganti dengan path ke tiga folder yang berisi file audio
folder_paths = ['datasets/happy', 'datasets/neutral', 'datasets/sad']

audio_paths = []
for folder_path in folder_paths:
    audio_files = os.listdir(folder_path)
    for audio_file in audio_files:
        audio_paths.append(os.path.join(folder_path, audio_file))

mfcc_extractor = MFCCExtractor(audio_paths)
mfccs = mfcc_extractor.extract_mfcc()

for i in range(len(audio_paths)):
    print(f'Audio {i+1}')
    print(f'Shape MFCCs: {mfccs[i].shape}')
    print('MFCCs:')
    print(mfccs[i])
    print('-------------------')
