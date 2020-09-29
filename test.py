import numpy as np
import glob
import os
from hparams import hparams
import pickle

root_dir = "/hd0/speechsplit/preprocessed/spmel"
feat_dir = "/hd0/speechsplit/preprocessed/raptf0"
file_name = "004/004_179.npy"

metadata = pickle.load(open('/hd0/speechsplit/preprocessed/spmel/train.pkl', "rb"))

sbmt_i = metadata[0]
emb_org = torch.from_numpy(sbmt_i[1]).to(device)

melsp = np.load(os.path.join(root_dir, file_name))
f0_org = np.load(os.path.join(feat_dir, file_name))

print(melsp[0:, :].shape)

len_crop = np.random.randint(hparams.min_len_seq, hparams.max_len_seq+1, size=2)
print(len(melsp) - len_crop)
left = np.random.randint(0, len(melsp)-len_crop, size=2)
print(left)

