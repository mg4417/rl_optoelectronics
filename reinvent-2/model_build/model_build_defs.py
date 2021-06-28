import sqlite3
import os
import pandas as pd
import numpy as np
import random
import pickle
import sklearn.ensemble
#from sklearn.metrics import roc_auc_score, mean_squared_error,mean_absolute_error

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, PandasTools
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.Draw import IPythonConsole
#from IPython.core.display import display, HTML

from shutil import copyfile

def connect_db(db_file, parameter):
    """
    Execute SQL query to obtain molecules and parameters
    Args:
        db_file: database for examination
        parameter: molecule parameter for query

    Returns: query result containing SMILES for molecules and their properties

    """
    #BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #db_path = os.path.join(BASE_DIR, db_file)
    db_path = db_file
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    t = (parameter,)
    c.execute("""SELECT text_key_values.key,text_key_values.value,text_key_values.id,
            number_key_values.key, number_key_values.value
            FROM text_key_values INNER JOIN number_key_values
            ON text_key_values.id=number_key_values.id 
            WHERE text_key_values.key='SMILES' and number_key_values.key=?""", t)
    result = c.fetchall()
    return result


def get_data(query_result):
    """
    Retrive lists of SMILES, compound unique ids and parameters from SQL query result
    Args:
        query_result: result for SQL query containing molecular information and property

    Returns: lists of smiles, compound uIDs and parameters

    """
    smiles = [item[1].rstrip() for item in query_result]
    compounds = [item[2] for item in query_result]
    gaps = [item[-1] for item in query_result]
    return smiles, compounds, gaps


def smiles_to_mols(query_smiles):
    mols = [Chem.MolFromSmiles(smile) for smile in query_smiles]
    valid = [0 if mol is None else 1 for mol in mols]
    valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
    valid_mols = [mols[idx] for idx in valid_idxs]
    return valid_mols, valid_idxs


class Descriptors:

    def __init__(self, data):
        self._data = data

    def ECFP(self, radius, nBits):
        fingerprints = []
        mols, idx = smiles_to_mols(self._data)
        fp_bits = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits) for mol in mols]
        for fp in fp_bits:
            fp_np = np.zeros((1, nBits), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return fingerprints, idx

    def ECFP_counts(self, radius, size, useFeatures, useCounts=True):
        mols, valid_idx = smiles_to_mols(self._data)
        fps = [AllChem.GetMorganFingerprint(mol, radius, useCounts=useCounts, useFeatures=useFeatures) for mol in mols]
        size = size
        nfp = np.zeros((len(fps), size), np.int32)
        for i, fp in enumerate(fps):
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % size
                nfp[i, nidx] += int(v)
        return nfp, valid_idx

    def Avalon(self, nBits):
        mols, valid_idx = smiles_to_mols(self._data)
        fingerprints = []
        fps = [pyAvalonTools.GetAvalonFP(mol, nBits=nBits) for mol in mols]
        for fp in fps:
            fp_np = np.zeros((1, nBits), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return fingerprints, valid_idx

    def MACCS_keys(self):
        mols, valid_idx = smiles_to_mols(self._data)
        fingerprints = []
        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in mols]
        for fp in fps:
            fp_np = np.zeros((1, ), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fp_np)
            fingerprints.append(fp_np)
        return fingerprints, valid_idx

def get_ECFP6_counts_backup(inp):
    if not isinstance(inp, list):
        inp = list(inp)
    desc = Descriptors(inp)
    fps, _ = desc.ECFP_counts(radius=3, size=1024, useFeatures=True, useCounts=True)
    return fps,_

def get_ECFP6_counts(inp,r,s):
    if not isinstance(inp, list):
        inp = list(inp)
    desc = Descriptors(inp)
    fps, _ = desc.ECFP_counts(radius=r, size=s, useFeatures=True, useCounts=True)
    return fps,_

def get_ECFP(inp):
    if not isinstance(inp, list):
        inp = list(inp)
    desc = Descriptors(inp)
    fps, _ = desc.ECFP(radius=3,nBits=1024)
    return fps,_

def get_MACCS_keys(inp):
    if not isinstance(inp, list):
        inp = list(inp)
    desc = Descriptors(inp)
    fps, _ = desc.MACCS_keys()
    return fps,_

def get_Avalon(inp):
    if not isinstance(inp, list):
        inp = list(inp)
    desc = Descriptors(inp)
    fps, _ = desc.Avalon(2048)
    return fps,_

