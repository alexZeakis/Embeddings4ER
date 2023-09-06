import pandas as pd
import os
import sys

#data_dir = '../../data/labeled/'
#data_dir_2 = '../../data/labeled_2/'
data_dir = sys.argv[1]
data_dir_2 = sys.argv[2]

def transform(df, tabA, tabB):
    temp_A = df['ltable_id'].apply(lambda x: tabA.loc[x])
    temp_A.columns = temp_A.columns.map(lambda x: 'left_'+x)
    temp_B = df['rtable_id'].apply(lambda x: tabB.loc[x])
    temp_B.columns = temp_B.columns.map(lambda x: 'right_'+x)    
    df = pd.concat([temp_A, temp_B, df['label']], axis=1)
    df['id'] = range(df.shape[0])
    return df


for idir in os.listdir(data_dir):
    if not os.path.isdir(data_dir+idir):
        continue
    if 'tableA.csv' not in os.listdir(data_dir+idir):
        continue
    print(idir)
    tabA = pd.read_csv(data_dir+idir+"/tableA.csv")
    tabB = pd.read_csv(data_dir+idir+"/tableB.csv")
    
    train = pd.read_csv(data_dir+idir+"/train.csv")
    val = pd.read_csv(data_dir+idir+"/valid.csv")
    test = pd.read_csv(data_dir+idir+"/test.csv")
    
    train = transform(train, tabA, tabB)
    val = transform(val, tabA, tabB)
    test = transform(test, tabA, tabB)
    
    if not os.path.exists(data_dir_2):
        os.mkdir(data_dir_2)
    if not os.path.exists(data_dir_2+idir):
        os.mkdir(data_dir_2+idir)
    train.to_csv(data_dir_2+idir+"/train.csv", header=True, index=False)
    val.to_csv(data_dir_2+idir+"/valid.csv", header=True, index=False)
    test.to_csv(data_dir_2+idir+"/test.csv", header=True, index=False)
