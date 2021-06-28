#import sample_from_model
grand_path = input('name of experiment folder: ')
steps = int(input('how many epochs? '))
how_many = int(input('how many SMILES to sample? '))
confirm = input(f'is "{grand_path} and {steps} steps correct? y/n: ')
if confirm=='y':
    nums = range(1,steps+1)
    for num in nums:
        input_path = f'../storage/{grand_path}/model.trained.{num}'
        output_path = f'../storage/{grand_path}/model.sampled.{num}'
        #sample_from_model.main(--model= input_path, -o= f'../storage/{grand_path}/logs/final_agent_samples_{num}.smi')
        print(f"python sample_from_model.py -m {input_path} -o {output_path} -n {how_many}")
    print('fin')