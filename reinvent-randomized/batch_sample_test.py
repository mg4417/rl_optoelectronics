import sample_from_model
import subprocess
import argparse
#import sys

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", "-e", help="Number of models (1 to n) to sample.", type=int, required=True)
parser.add_argument("--folder", "-f", help="Name of target folder.", type=str, required=True)
parser.add_argument("--samples", "-s", help="Number of samples to generate per model.", type=int, required=True)

args = parser.parse_args()

for i in range(1,args.epochs+1):
    #sys.argv = ['-m', f'../storage/rand_3/model.trained.{i}', '-o', f'../storage/rand_3/model.sampled.{i}', '-n', '10']
    #sample_from_model.main()
    subprocess.call(["python", "sample_from_model.py", '-m', f'../storage/{args.folder}/model.trained.{i}', '-o', f'../storage/{args.folder}/model.sampled.{i}', '-n', f'{args.samples}'])