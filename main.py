import IPython.display as ipd
import librosa.display
import numpy as np
import pandas as pd
from datetime import datetime

# Sources:
# https://www.youtube.com/watch?v=WJI-17MNpdE

start_time = datetime.now()

df = pd.read_csv('Dataset/participant_urbansound8k.csv')

labels = df.iloc[:, 3]
files = df.iloc[:, 0]
folder = df.iloc[:, 2]


for row in range(7079):
    audio_file = f"Dataset/fold{folder[row]}/{files[row]}"
    ipd.Audio(audio_file)
    signal, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
    deltaMfccs = librosa.feature.delta(mfccs)
    delta2Mfccs = librosa.feature.delta(mfccs, order=2)
    compmfccs = np.concatenate((mfccs, deltaMfccs, delta2Mfccs))
    print(compmfccs)

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))


