import torch
import pickle
import numpy as np
from hparams import hparams
from utils import pad_seq_to_2
from utils import quantize_f0_numpy
from model import Generator_3 as Generator
from model import Generator_6 as F0_Converter
import matplotlib.pyplot as plt
import os
import glob

out_len = 408

device = 'cuda:1'
G = Generator(hparams).eval().to(device)
g_checkpoint = torch.load('run/models/234000-G.ckpt', map_location=lambda storage, loc: storage)
G.load_state_dict(g_checkpoint['model'])

metadata = pickle.load(open('/hd0/speechsplit/preprocessed/spmel/train.pkl', "rb"))

sbmt_i = metadata[0]
emb_org = torch.from_numpy(sbmt_i[1]).unsqueeze(0).to(device)

root_dir = "/hd0/speechsplit/preprocessed/spmel"
feat_dir = "/hd0/speechsplit/preprocessed/raptf0"

# mel-spectrogram, f0 contour load
x_org = np.load(os.path.join(root_dir, sbmt_i[2]))
f0_org = np.load(os.path.join(feat_dir, sbmt_i[2]))
len_org = x_org.shape[0]

uttr_org_pad, len_org_pad = pad_seq_to_2(x_org[np.newaxis,:,:], out_len)
uttr_org_pad = torch.from_numpy(uttr_org_pad).to(device)
f0_org_pad = np.pad(f0_org, (0, out_len-len_org), 'constant', constant_values=(0, 0))
f0_org_quantized = quantize_f0_numpy(f0_org_pad)[0]
f0_org_onehot = f0_org_quantized[np.newaxis, :, :]
f0_org_onehot = torch.from_numpy(f0_org_onehot).to(device)
uttr_f0_org = torch.cat((uttr_org_pad, f0_org_onehot), dim=-1)

sbmt_j = metadata[1]
emb_trg = torch.from_numpy(sbmt_j[1]).unsqueeze(0).to(device)
#x_trg, f0_trg, len_trg, uid_trg = sbmt_j[2]
x_trg = np.load(os.path.join(root_dir, sbmt_j[2]))
f0_trg = np.load(os.path.join(feat_dir, sbmt_j[2]))
len_trg = x_org.shape[0]

# 지금은 안씀
# uttr_trg_pad, len_trg_pad = pad_seq_to_2(x_trg[np.newaxis,:,:], out_len)
# uttr_trg_pad = torch.from_numpy(uttr_trg_pad).to(device)
# f0_trg_pad = np.pad(f0_trg, (0, out_len-len_trg), 'constant', constant_values=(0, 0))
# f0_trg_quantized = quantize_f0_numpy(f0_trg_pad)[0]
# f0_trg_onehot = f0_trg_quantized[np.newaxis, :, :]
# f0_trg_onehot = torch.from_numpy(f0_trg_onehot).to(device)

x_identic_val = G(uttr_f0_org, uttr_org_pad, emb_org)

output = x_identic_val.squeeze().cpu().detach().numpy()
plt.imshow(output.T)
plt.show()
plt.gca().invert_yaxis()