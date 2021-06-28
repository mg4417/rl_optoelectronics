import moses
import os
import numpy as np
import logging, sys
logging.disable(sys.maxsize)
import warnings
warnings.filterwarnings('ignore')

def smiles_to_list(smiles_file):
    with open(smiles_file) as f:
        content = f.readlines()
    out_list = [x.strip() for x in content] 
    return out_list

#train_smiles = os.path.expanduser("~/reinvent-2/data/model1_cano.smi")
train_smiles = '/Volumes/mg4417/home/reinvent-2/data/model1_cano.smi'
train_list = smiles_to_list(train_smiles)

#sampled_dir = os.path.expanduser("~/reinvent-2/outputs/REINVENT_transfer_learning_demo/d_075/sampled")
sampled_dir = '/Volumes/mg4417/home/reinvent-2/outputs/REINVENT_transfer_learning_demo/sampled'
m_path = os.path.join(sampled_dir,'100.smi')
l_gen = smiles_to_list(m_path)

metrics = moses.get_all_metrics(l_gen, k=[100,1000], test=train_list, train=train_list, n_jobs = 4)
print(metrics)