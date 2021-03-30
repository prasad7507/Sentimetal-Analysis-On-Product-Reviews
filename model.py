import numpy as np
import pandas as pd
df=pd.read_csv('D:/Sentimental Analysis/balanced_reviews.csv')
df.dropna(inplace=True)
df=df[df['overall']!=3]
#print(df['overall'].value_counts())
df['Positivity']=np.where(df['overall']>3,1,0)
#print(df['Positivity'].value_counts())
from sklearn.model_selection import train_test_split
ftrain,ftest,ltrain,ltest=train_test_split(df['reviewText'], df['Positivity'],random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer
vect=TfidfVectorizer(min_df=5).fit(ftrain)
vect.vocabulary_
#print(vect.get_feature_names()[10000:10010])
ftrain_vectorized=vect.transform(ftrain)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(ftrain_vectorized,ltrain)
predictions=model.predict(vect.transform(ftest))
from sklearn.metrics import confusion_matrix
confusion_matrix(ltest,predictions)
from sklearn.metrics import roc_auc_score
print(roc_auc_score(ltest,predictions))

import pickle
pkl_file='pickle_model.pkl'
file=open(pkl_file,'wb')
pickle.dump(model,file)
file.close()

pickle.dump(vect.vocabulary_,open('feature.plk','wb'))
