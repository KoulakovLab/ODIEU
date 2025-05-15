import pandas as pd

model: str = "NovaSearch.stella_en_400M_v5"

dfs = list()
for fold in range(1, 6):
    df = pd.read_pickle(f"{model}_{fold}.pkl")
    dfs.append(df[df.testset])
df_testset = pd.concat(dfs)
df_testset
df_testset.testset.sum()
df_testset.to_csv(f"{model}_testset_only.csv", sep="$")
pd.to_pickle(df_testset, f"{model}_testset_only.pkl")
