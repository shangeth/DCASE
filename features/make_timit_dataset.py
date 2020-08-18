import kaldiio
from glob import glob
import numpy as np
import pandas as pd
import os

label_dir = 'timit_labels/'
label_files = ['utt2height_trainNet_sp', 'utt2height_valid',
                'utt2height_test']
label_files = [label_dir + x for x in label_files]

data_dir = 'dataset/'
ark_dir = ['trainNet_sp/deltafalse', 'valid/deltafalse', 'test/deltafalse']
ark_dir = [data_dir+x for x in ark_dir]
 
save_dir = 'data/timit_dataset/'
save_sub_dirs = ['train/', 'valid/', 'test/']
save_sub_dirs = [save_dir+x for x in save_sub_dirs]

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i in save_sub_dirs:
    if not os.path.exists(i):
        os.makedirs(i)

for i, data_type in enumerate(ark_dir):
    dir = data_type
    save_path = save_sub_dirs[i]
    label_file = label_files[i]
    df = pd.read_csv(label_file, sep='\t', header=None)
    df.set_index(0, inplace=True)
    for file in glob(f'{dir}/*.ark'):
        d = kaldiio.load_ark(file)
        for key, numpy_array in d:
            label = df[1][key]
            print(f'{save_path}{key}_{label}.npy')
            np.save(f'{save_path}{key}_{label}.npy', numpy_array.T)
    
for i in save_sub_dirs:
    datatype = i.split('/')[1]
    num = len(os.listdir(i))
    print(f'{datatype} => {num}')