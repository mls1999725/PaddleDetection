import pickle

f = open("ppyolo.pdparams")
model = pickle.load(f)
print(model.keys())
f.close()
