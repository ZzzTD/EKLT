import pandas as pd
import numpy as np
from PyEMD import EMD,Visualisation
import os
import matplotlib.pyplot as plt

data = pd.read_csv('../data/H_river_waterlevel.csv')
datas = data['LZ_Z']
dfs = datas.values

emd = EMD()
IMFs = emd(dfs)
vis = Visualisation(emd)
vis.plot_imfs()
vis.show()

full_imf = pd.DataFrame(IMFs)
emd = full_imf.T

emd.to_excel('../data/emd.xlsx', index=False)

output_folder = "../data/emd/"
os.makedirs(output_folder, exist_ok=True)

for i, col in enumerate(emd.columns, 1):
    new_data_table = data.copy()
    new_data_table['LZ_Z'] = emd[col].values
    output_file_path = os.path.join(output_folder, f"emd{i}.csv")
    new_data_table.to_csv(output_file_path, index=False)
    print(f"New Data Table {i} saved to {output_file_path}")

