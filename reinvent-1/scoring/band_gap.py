import logging
import pickle
from typing import List

#from sklearn.externals import joblib
import joblib

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import utils

class band_gap(object):
    """Scores 1 for valid molecules with a HOMO-LUMO gap < 2 eV, and 0 otherwise."""

    def __init__(self, predictor_path: str, zero_score = 0.0, full_score = 1.0):
        self.predictor_path = predictor_path
        self.model = joblib.load(predictor_path)
        self.zero_score = zero_score
        self.full_score = full_score

    def __call__(self, smiles: List[str]) -> dict:
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        valid = [1 if mol is not None else 0 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]

        fps = band_gap.fingerprints_from_mols(valid_mols)
        #predictions = band_gap.predict_property(self.predictor_path, fps)
        predictions = self.model.predict(fps)
        less_two = [self.full_score if pd<2 else self.zero_score for pd in predictions]
        
        score = np.full(len(smiles), 0, dtype=np.float32)
        for idx, value in zip(valid_idxs, less_two):
            score[idx] = value
        return {"total_score": np.array(score, dtype=np.float32)}

    @classmethod
    def predict_property(cls, model_file, fps):
        """
        Function to predict the properties of generated molecules
        Args:
            model_file: File containing pre-trained ML model for prediction
            fps: list of molecular fingerprints
        Returns: list of predicted values
        """
        model = joblib.load(model_file)
        return model.predict(fps)
    
    @classmethod
    def fingerprints_from_mols(cls, mols):
        fps = [AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=False) for mol in mols]
        size = 1024
        nfp = np.zeros((len(fps), size), np.int32)
        for i, fp in enumerate(fps):
            for idx, v in fp.GetNonzeroElements().items():
                nidx = idx % size
                nfp[i, nidx] += int(v)
        return nfp