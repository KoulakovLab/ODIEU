import sys

import numpy as np
import pandas as pd
import scipy.io
from tqdm import tqdm














rng = np.random.default_rng(93825320)

model_paths: list[list[str]] = [[sys.argv[1] + f"/sbert_init" for _ in range(5)], [sys.argv[1] + f"/sbert_best_left_{fold+1}" for fold in range(5)]]
model_names: list[list[str]] = [[path.replace("/", ".") for path in list_paths] for list_paths in model_paths]

description_types = list(pd.read_csv("description_types.csv", names=["Labels"])['Labels'])
contenders = pd.read_pickle("descriptions.pkl")

for i_model in tqdm(range(len(model_paths))):
    for fold in tqdm(range(5)):
        model_path = model_paths[i_model][fold]
        model_name = model_names[i_model][fold]
        
        testset = pd.read_pickle(f"testsets_{fold+1}.pkl")
        embeddings = pd.read_pickle(f"embeddings_{model_name}_fold_{fold+1}.pkl")
        embedding_size = np.array(embeddings.loc[:, 'Beautified human descriptions'].to_list()).shape[1]
        cosine_embeddings = {contender_name: [np.zeros((0, embedding_size)), np.zeros((0, embedding_size)), np.zeros((0, embedding_size))] for contender_name in description_types[1:]}
        for contender_name in description_types[1:]:
            for dataset_pair in [["arctander", "goodscents"], ["goodscents", "leffingwell"], ["leffingwell", "arctander"]]:
                cids_intersection = testset[testset.Dataset == dataset_pair[0]].index.intersection(embeddings[embeddings.Dataset == dataset_pair[1]].index)
                a0 = np.array(embeddings[embeddings.Dataset == dataset_pair[0]].loc[cids_intersection, description_types[1]].to_list())
                a1 = np.array(embeddings[embeddings.Dataset == dataset_pair[1]].loc[cids_intersection, contender_name].to_list())
                cosine_embeddings[contender_name][0] = np.concatenate((cosine_embeddings[contender_name][0], a0))
                cosine_embeddings[contender_name][1] = np.concatenate((cosine_embeddings[contender_name][1], a1))
                cosine_embeddings[contender_name][2] = np.concatenate((cosine_embeddings[contender_name][2], rng.permutation(a1)))
        df_cosine_embeddings = pd.DataFrame.from_dict(cosine_embeddings)
        df_cosine_embeddings.to_csv(f"cosine_embeddings_{model_name}_fold_{fold+1}.csv")
        pd.to_pickle(df_cosine_embeddings, f"cosine_embeddings_{model_name}_fold_{fold+1}.pkl")
        scipy.io.savemat(f"cosine_embeddings_{model_name}_fold_{fold+1}.mat", cosine_embeddings)

for i_model in tqdm(range(len(model_paths))):
    for fold in tqdm(range(5)):
        model_path = model_paths[i_model][fold]
        model_name = model_names[i_model][fold]
        
        testset = pd.read_pickle(f"testsets_{fold+1}.pkl")
        embeddings = pd.read_pickle(f"embeddings_{model_name}_fold_{fold+1}.pkl")
        embedding_size = np.array(embeddings.loc[:, 'Beautified human descriptions'].to_list()).shape[1]
        cosine_embeddings = {contender_name: [np.zeros((0, embedding_size)), np.zeros((0, embedding_size)), np.zeros((0, embedding_size))] for contender_name in description_types[1:]}
        for contender_name in description_types[1:]:
            for dataset_pair in [["arctander", "goodscents"], ["goodscents", "leffingwell"], ["leffingwell", "arctander"]]:
                cids_intersection = testset[testset.Dataset == dataset_pair[0]].index.intersection(embeddings[embeddings.Dataset == dataset_pair[1]].index)
                a0 = np.array(embeddings[embeddings.Dataset == dataset_pair[0]].loc[cids_intersection, description_types[1]].to_list())
                b0 = np.array(embeddings[embeddings.Dataset == dataset_pair[1]].loc[cids_intersection, description_types[1]].to_list())
                a1 = np.array(embeddings[embeddings.Dataset == dataset_pair[1]].loc[cids_intersection, contender_name].to_list())
                b1 = np.array(embeddings[embeddings.Dataset == dataset_pair[0]].loc[cids_intersection, contender_name].to_list())
                cosine_embeddings[contender_name][0] = np.concatenate((cosine_embeddings[contender_name][0], a0, b0))
                cosine_embeddings[contender_name][1] = np.concatenate((cosine_embeddings[contender_name][1], a1, b1))
                cosine_embeddings[contender_name][2] = np.concatenate((cosine_embeddings[contender_name][2], rng.permutation(a1), rng.permutation(b1)))
        df_cosine_embeddings = pd.DataFrame.from_dict(cosine_embeddings)
        df_cosine_embeddings.to_csv(f"cosine_embeddings_symmetric_{model_name}_fold_{fold+1}.csv")
        pd.to_pickle(df_cosine_embeddings, f"cosine_embeddings_symmetric_{model_name}_fold_{fold+1}.pkl")
        scipy.io.savemat(f"cosine_embeddings_symmetric_{model_name}_fold_{fold+1}.mat", cosine_embeddings)
