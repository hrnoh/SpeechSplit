import pickle

spk2gen_file = "assets/spk2gen.pkl"

with open("spk_info.txt", 'rt') as f:
    spkinfo = [l.rstrip() for l in f.readlines()]

spk2gen = {}
for info in spkinfo:
    spk, gender = info.split()
    spk2gen[spk] = gender

with open(spk2gen_file, 'wb') as f:
    pickle.dump(spk2gen, f)