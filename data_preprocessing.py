import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from fastai.tabular.all import df_shrink
to_clean = '/kaggle/input/unswnb15/UNSW_NB15_training-set.csv'
df = pd.read_csv(to_clean, skipinitialspace=True, low_memory=False, encoding='utf-8')
df.drop(columns=['id'], inplace=True)
df.dtypes
df = df_shrink(df, skip=[], obj2cat=True, int2uint=False)
df.dtypes
df.isna().sum()

df.to_parquet(f"/kaggle/working/{to_clean.split('/')[-1].replace('.csv', '.parquet')}")
!ls -lth /kaggle/input/unswnb15
!ls -lth /kaggle/working


