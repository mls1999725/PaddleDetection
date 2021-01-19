import os
import pickle

models_dir = "output/ppyolo_pan"
starting_id = 450000
ending_id = 457200
model_names = list(range(starting_id, ending_id, 600))
#model_names = os.listdir(models_dir)
#model_names = [f for f in model_names if f.endswith('.pdparams')]
model_dirs = [
    os.path.join(models_dir, str(i) + '.pdparams')
    #os.path.join(models_dir, i)
    for i in model_names
]
models = [pickle.load(open(model_dir, 'rb')) for model_dir in model_dirs]
model_num = len(models)
model_keys = models[-1].keys()
new_model = models[-1].copy()

for key in model_keys:
    sum_weight = 0.0
    for m in models:
        sum_weight += m[key]
    avg_weight = sum_weight / model_num
    new_model[key] = avg_weight
save_model_name = 'swa_' + str(starting_id) + '-' + str(ending_id) + '.pdparams'
#save_model_name = 'swa.pdparams'
save_dir = os.path.join(models_dir, save_model_name)
pickle.dump(new_model, open(save_dir, 'wb'), protocol=2)
    
