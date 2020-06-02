import numpy as np 
import pandas as pd

test_labels = pd.read_csv('../../data/raw/test_labels.csv')
test = pd.read_csv('../../data/raw/test.csv')
train = pd.read_csv('../../data/raw/train.csv')

#We are going to merge train and test and randomly sample a new test later in another script.
#This is because test from competition comes from different distribution than train.
#This causes many problems which are not the focus of this project.

#Test set given in competition have some rows with -1 which where not used for scoring
#And we don't know the ground truth, so we must filter them out.
mask = (test_labels.drop('id',axis=1)!=-1).all(axis=1)
test_labels = test_labels[mask]
test = test.merge(test_labels,on='id')


#Concat train and test
df = pd.concat([train,test])
df = df.sample(frac=1)
df.to_csv('../../data/raw/dataset.csv',index = False)

