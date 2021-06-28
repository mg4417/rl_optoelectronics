import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import sqlite3
#import os
import pandas as pd
import numpy as np
import random
import pickle
import sklearn.ensemble
from sklearn.metrics import roc_auc_score, mean_squared_error,mean_absolute_error
import sklearn.model_selection
import sklearn.svm

import argparse

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, PandasTools
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.Draw import IPythonConsole
from IPython.core.display import display, HTML

from model_build_defs import *

import optuna
from xgboost import XGBRegressor

from shutil import copyfile

def objective(trial):
    
    if model_type == 'xgb':
        learning_rate = trial.suggest_float('learning_rate', 0.03, 0.3)
        subsample = trial.suggest_float('subsample', 0.7, 0.9)
        n_estimators = 400
        max_depth = trial.suggest_int('max_depth', 4, 7)    
        #min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [20,30,40])

        regressor_obj = XGBRegressor(learning_rate=learning_rate,
                                     subsample=subsample,
                                     n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     n_jobs=1,
                                     #max_features="auto",
                                     #min_samples_leaf=min_samples_leaf,
                                    )
        
    if model_type == 'gbdt':
        learning_rate = trial.suggest_float('learning_rate', 0.03, 0.3)
        subsample = trial.suggest_float('subsample', 0.7, 0.9)
        n_estimators = 400
        max_depth = trial.suggest_int('max_depth', 4, 7)    
        min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [20,30,40])
        
        regressor_obj = sklearn.ensemble.GradientBoostingRegressor(learning_rate=learning_rate,
                                                                   subsample=subsample,
                                                                   n_estimators=n_estimators,
                                                                   max_depth=max_depth,
                                                                   #max_features="auto",
                                                                   min_samples_leaf=min_samples_leaf,
                                                                  )

    if model_type == 'rf':
        n_estimators = 400
        max_depth = trial.suggest_categorical('max_depth', [15,25,50,100,None])
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=max_depth,
                                                               max_features="auto",
                                                               n_estimators=n_estimators,
                                                               n_jobs=1,
                                                               #class_weight="balanced"
                                                              )
    
    if model_type == 'svr':
        C = trial.suggest_float('C', 0.1, 1000.0)
        epsilon = trial.suggest_float('epsilon', 0.0001, 10.0)
        gamma = trial.suggest_float('gamma', 0.0001, 5.0)
        regressor_obj = sklearn.svm.SVR(C=C,
                                       epsilon=epsilon,
                                       gamma=gamma,)
        
        
        
    score = sklearn.model_selection.cross_val_score(regressor_obj, fps_train, gaps_train, n_jobs= n_cpus, cv=4, verbose=10)
    accuracy = score.mean()
    return accuracy

    #regressor_obj.fit(all_fps, all_gaps)
    #y_pred_final = regressor_obj.predict(X=all_fps)
    #train_error = mean_absolute_error(y_true=all_gaps, y_pred=y_pred_final)
    #return train_error
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Specify property and output path')
    parser.add_argument('--fingerprint', '-fp', type=str, required=True)
    parser.add_argument('--model_type', '-m', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--property', '-p', type=str, required=True)
    parser.add_argument('--trials', '-t', type=int, required=True)
    parser.add_argument('--cpu', '-c', type=int, required=True)
    args = parser.parse_args()
    model_type = args.model_type
    n_cpus = args.cpu
    fingerprint = args.fingerprint
    
    # --------- change these path variables as required
    output_dir = os.path.expanduser("~/reinvent-2/outputs/REINVENT_model_building_demo")

    # --------- do not change
    # get the notebook's root path
    try: ipynb_path
    except NameError: ipynb_path = os.getcwd()

    # if required, generate a folder to store the results
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    #qresult = connect_db('solar.db', f'{args.property}') #change this
    #smiles, compounds, gaps = get_data(qresult)
    
    dft_dir = os.path.expanduser("~/predictor-train/dft_result.csv")
    df = pd.read_csv(dft_dir)

    smiles = df['SMILES'].tolist()
    compounds = df['id'].tolist()
    if args.property == 'KS_gap':
        gaps = df['first_triplet'].tolist()
    if args.property == 'dip':
        gaps = df['dipole_moment'].tolist()
    
    if fingerprint == 'ecfp6_3_1024':
        all_fps, index = get_ECFP6_counts(smiles,3,1024)
    if fingerprint == 'ecfp6_4_1024':
        all_fps, index = get_ECFP6_counts(smiles,4,1024)
    if fingerprint == 'ecfp6_3_2048':
        all_fps, index = get_ECFP6_counts(smiles,3,2048)
    if fingerprint == 'ecfp6_4_2048':
        all_fps, index = get_ECFP6_counts(smiles,4,2048)
    if fingerprint == 'maccs':
        all_fps, index = get_MACCS_keys(smiles)
    if fingerprint == 'avalon':
        all_fps, index = get_Avalon(smiles)
    
    all_gaps = list(gaps[i] for i in index) 
    fps_train, fps_test, gaps_train, gaps_test = sklearn.model_selection.train_test_split(all_fps, all_gaps, test_size=0.20, random_state=42)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials= args.trials, n_jobs=1)
    trial=study.best_trial
    print(trial)
    print('----------------------')
    regressor_final = XGBRegressor(learning_rate=trial.params['learning_rate'],
                                   subsample=trial.params['subsample'],
                                   n_estimators=400,
                                   max_depth=trial.params['max_depth'],
                                   n_jobs=1,
                                   #max_features="auto",
                                   #min_samples_leaf=min_samples_leaf,
                                  )
    regressor_final.fit(fps_train, gaps_train)
    y_pred_final = regressor_final.predict(fps_test)
    mae = mean_absolute_error(y_true=gaps_test, y_pred=y_pred_final)
    #train_score = mean_squared_error(y_true=all_gaps, y_pred=y_pred_final)
    print(f"   final model MAE: {mae}")
    # save the model
    with open(os.path.join(output_dir, f"{args.output}.pkl"), "wb") as f:
        pickle.dump(regressor_final, f)