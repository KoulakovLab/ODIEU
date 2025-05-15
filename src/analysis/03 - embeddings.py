import sys

import numpy as np
import pandas as pd
import scipy.io
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

descriptions = pd.read_pickle("descriptions.pkl")

model_paths: list[str] = [sys.argv[1] + "/sbert_init" for _ in range(5)]
model_paths.extend([sys.argv[1] + f"/sbert_best_left_{fold+1}" for fold in range(5)])
model_names: list[str] = [path.replace("/", ".") for path in model_paths]

for i_model in range(len(model_paths)):
    for fold in tqdm(range(5)):
        model_path = model_paths[i_model]
        model_name = model_names[i_model]
        
        embeddings = pd.DataFrame(data=[], columns=descriptions.columns)
        for column in ['CID', 'Dataset']:
            embeddings[column] = descriptions[column]
            
        aggregateset = np.load(f"datasets/cross_validation/fold_{fold+1}.npz")
        testsets_datasets = np.array([s.replace("human/", "") for s in aggregateset['testset_datasets']])
        is_in_testset = (descriptions.CID == -1)
        for icid, cid in enumerate(aggregateset['testset_cids']):
            is_in_testset |= ((descriptions.CID == cid) & (descriptions.Dataset == testsets_datasets[icid]))
        embeddings['testset'] = is_in_testset.to_numpy()
        
        embeddings.rename(columns={'Description': 'Human descriptions',
                                   'BagOfWords': 'Human labels',
                                   'generated_human': 'Beautified human descriptions',
                                   'generated_w_labels': 'Beautified human labels',
                                   'NEW_deepnose_sentences': 'Beautified DeepNose descriptions',
                                   'NEW_deepnose_labels': 'Beautified DeepNose labels',
                                   'zero_shot': 'From SMILES - zero shot',
                                   'few_shot_human': 'From SMILES - human examples',
                                   'few_shot_gen_BoW': 'From SMILES - beautified human examples',
                                   'few_shot_gen_human': 'From SMILES - beautified labels examples'},
                          inplace=True)
        for dataset in ["arctander", "goodscents", "leffingwell"]:
            embeddings[embeddings.Dataset == dataset] = embeddings[embeddings.Dataset == dataset].drop_duplicates(subset="CID", keep=False)
            pd.DataFrame.drop_duplicates
        embeddings.set_index('CID', inplace=True)
        
        sbert = SentenceTransformer(f"finetuning/{model_path}", trust_remote_code=True, config_kwargs={'use_memory_efficient_attention': True, 'unpad_inputs': True})
        for column in ['Beautified human descriptions', 'Human descriptions', 'Human labels', 'Beautified human labels', 'Beautified DeepNose descriptions','Beautified DeepNose labels', 'From SMILES - zero shot', 'From SMILES - human examples', 'From SMILES - beautified human examples', 'From SMILES - beautified labels examples']:
            embeddings[column] = sbert.encode(np.array(descriptions[column].to_list())).tolist()
        
        description_types = ['Dataset', 'Beautified human descriptions', 'Human descriptions', 'Human labels', 'Beautified human labels', 'Beautified DeepNose descriptions','Beautified DeepNose labels', 'From SMILES - zero shot', 'From SMILES - human examples', 'From SMILES - beautified human examples', 'From SMILES - beautified labels examples']
        embeddings = embeddings[description_types]
        embeddings.to_csv(f"{model_name}_fold_{fold+1}.csv", sep="$")
        pd.to_pickle(embeddings, f"{model_name}_fold_{fold+1}.pkl")
        scipy.io.savemat(f"{model_name}_fold_{fold+1}.mat", embeddings.to_dict())
