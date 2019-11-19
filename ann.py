import os, sys
import pandas as pd
name = sys.argv[1]
df = pd.read_csv(name)
df['ours'] = df['class'].apply(lambda x: None)

for i, row in enumerate(df.values):
    print('{}/{} '.format(i, len(df)), row[1], row[2])
    class_id = int(input())
    df.loc[i,'ours'] = class_id
    break
df.to_csv('out_'+name)