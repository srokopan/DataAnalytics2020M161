from datasketch import MinHash, MinHashLSH
import pandas as pd
import re
import timeit

#start timer
start = timeit.default_timer()


def preprocess(text):
    text = re.sub(r'[^\w\s]','',text)
    tokens = text.lower()
    tokens = tokens.split()
    return tokens


train=pd.read_csv("corpusTrain.csv",dtype=str).apply(lambda x: x.astype(str).str.lower())
test=pd.read_csv("corpusTest.csv",dtype=str).apply(lambda x: x.astype(str).str.lower())

data=train['Content'].tolist()
datatest=test['Content'].tolist()


# Create an MinHashLSH index optimized for Jaccard threshold 0.8,
# that accepts MinHash objects with 128 permutations functions
lsh = MinHashLSH(threshold=0.8, num_perm=16)

# Create MinHash objects
minhashes = {}
for c, i in enumerate(data):
  minhash = MinHash(num_perm=16)

  for d in data[c]:
      #preprocess our data
        tokens = preprocess(d)
        for s in tokens:
          #minhshupdate
          minhash.update("".join(d).encode('utf-8'))
   #insert in lsh       
  lsh.insert(c, minhash)


for c,i in enumerate(datatest):
    m1 = MinHash(num_perm=16)
    for d in datatest[c]:
        tokens = preprocess(d)
        for s in tokens:
          m1.update("".join(d).encode('utf-8'))
    minhashes[c] = m1
 

#finished building
build = timeit.default_timer()
print('Build Time: ', build - start)  


duplicates=0
for i in range(len(minhashes.keys())):
  result = lsh.query(minhashes[i])
  if(len(result)>0):
      duplicates+=1

#finished querying
queryTime = timeit.default_timer()
print('Query Time: ', queryTime - build)  

print('number of duplicates are: ',duplicates)