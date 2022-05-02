import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import timeit

#start timer
start = timeit.default_timer()

#run on chunks due to ram problem (even google collab crashes)
chunk_size = 500 

train=pd.read_csv("corpusTrain.csv")
test=pd.read_csv("corpusTest.csv")
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(train['Content'])
tfidf_matrix2 = tfidf_vectorizer.transform(test['Content'])
matrix_len=tfidf_matrix2.shape[0]


def similarity_cosine_by_chunk(start, end):
    if end > matrix_len:
        end = matrix_len
    return cosine_similarity(tfidf_matrix2[start:end], tfidf_matrix) # scikit-learn function


total=0
for chunk_start in range(0, matrix_len, chunk_size):
  X=similarity_cosine_by_chunk(chunk_start, chunk_start+chunk_size)
  #print((X>0.8).sum())
  if((X>0.8).sum()):
    #calculate duplicates
    total+=(X>0.8).sum()
print(total)
#stop timer
stop = timeit.default_timer()
#print total time

print('Time: ', stop - start)  
