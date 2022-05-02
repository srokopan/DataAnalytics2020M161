import pandas as pd
import numpy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
import timeit


#start timer
start = timeit.default_timer()

train=pd.read_csv("corpusTrain.csv")
test=pd.read_csv("corpusTest.csv")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train['Content'])
Y = vectorizer.transform(test['Content'])

arr = X.toarray()
arr2=Y.toarray()
duplicates=0
for y in range(0,len(arr2)):
    for x in range(0,len(arr)):
        score=jaccard_score(arr2[y], arr[x],average="macro") 
        if(score>0.8):
            duplicates+=1

print(duplicates)
stop = timeit.default_timer()
print('Time: ', stop - start)  
