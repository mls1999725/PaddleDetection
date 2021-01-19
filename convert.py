import pickle
import numpy as np
import torch
import paddle

dst = {}
src = torch.load("./weights/yolov5x.pt")
print(src['model'].state_dict().keys())
for k, v in src['model'].state_dict().items():
    dst[k] = v.cpu().numpy()
    if dst[k].dtype == np.float16:
        print("1111")
        dst[k] = dst[k].astype(np.float32)
    print(dst[k].dtype)
pickle.dump(dst, open("./weights/yolov5x.pdparams", 'wb'), protocol=2)
