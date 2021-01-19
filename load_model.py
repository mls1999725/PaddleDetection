import pickle

weight_dict = pickle.load(open("./yolov5x.pdparams", 'rb'))
print(type(weight_dict))
