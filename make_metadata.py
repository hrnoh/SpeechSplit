import os
import pickle
import numpy as np
from hparams import hparams

#rootDir = 'assets/spmel'
rootDir = '/hd0/speechsplit/preprocessed/spmel'
pitchDir = '/hd0/speechsplit/preprocessed/raptf0'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


speakers = []
for idx, speaker in enumerate(sorted(subdirList)):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    
    # use hardcoded onehot embeddings in order to be cosistent with the test speakers
    # modify as needed
    # may use generalized speaker embedding for zero-shot conversion
    spkid = np.zeros((20,), dtype=np.float32)
    spkid[idx] = 1.0
    # if speaker == 'p226':
    #     spkid[1] = 1.0
    # else:
    #     spkid[7] = 1.0
    utterances.append(spkid)
    
    # create file list
    for fileName in sorted(fileList)[:-1]:
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)
    
with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)

# validation
speakers = []
for idx, speaker in enumerate(sorted(subdirList)):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))

    # use hardcoded onehot embeddings in order to be cosistent with the test speakers
    # modify as needed
    # may use generalized speaker embedding for zero-shot conversion
    spkid = np.zeros((20,), dtype=np.float32)
    spkid[idx] = 1.0
    # if speaker == 'p226':
    #     spkid[1] = 1.0
    # else:
    #     spkid[7] = 1.0
    utterances.append(spkid)

    # create file list
    mel = np.load(os.path.join(rootDir, speaker, sorted(fileList)[-1]))
    mel_len = mel.shape[0]
    f0 = np.load(os.path.join(pitchDir, speaker, sorted(fileList)[-1]))
    pitch_len = f0.shape[0]
    if mel_len > hparams.max_len_pad:
        utterances.append([mel[:hparams.max_len_pad], f0[:hparams.max_len_pad], hparams.max_len_pad, hparams.max_len_pad])
    else:
        utterances.append([mel, f0, mel_len, pitch_len])

    speakers.append(utterances)

with open(os.path.join(rootDir, 'val.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)