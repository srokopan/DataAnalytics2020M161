import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from subprocess import check_output
import csv
import os
import gc
import re
from nltk.stem import PorterStemmer
import re
from nltk.corpus import stopwords
import distance
from fuzzywuzzy import fuzz
import sys
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import re
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.model_selection import KFold, cross_val_score,cross_validate


from os import path


##########################GENERAL FUNCTIONS FOR EXTRACTING ADVANCED FEATURES

#set safe_div for feature extraction
SAFE_DIV = 0.0001

STOP_WORDS = stopwords.words("english")
  

def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)

    porter = PorterStemmer()
    pattern = re.compile('\W')

    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)

    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()

    return x


def get_token_features(q1, q2):
    token_features = [0.0] * 10

    # Converting the Sentence into Tokens:
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    # Get the non-stopwords in Questions
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    # Get the stopwords in Questions
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))

    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))

    # Get the common Tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    # Last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])

    # First word of both question is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))

    # Average Token Length of both Questions
    token_features[9] = (len(q1_tokens) + len(q2_tokens)) / 2
    return token_features


# get the Longest Common sub string

def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)


def extract_features(df):
    # extract features
        df['q1len'] = df['Question1'].str.len()
        df['q2len'] = df['Question2'].str.len()
        df['q1_n_words'] = df['Question1'].apply(lambda row: len(row.split(" ")))
        df['q2_n_words'] = df['Question2'].apply(lambda row: len(row.split(" ")))


        df['word_Common'] = df.apply(normalized_word_Common, axis=1)


        df['word_Total'] = df.apply(normalized_word_Total, axis=1)


        df['word_share'] = df.apply(normalized_word_share, axis=1)

        df['freq_q1+q2'] = df['freq_qid1']+df['freq_qid2']
        df['freq_q1-q2'] = abs(df['freq_qid1']-df['freq_qid2'])


    return df

########CODE STARTS HERE

#############LOAD THE TRAIN DATA AND EXTRACT THE FEATURES

print("Extracting features for train:")
df = pd.read_csv("train.csv", encoding='latin-1')
df = df.fillna('')
print(df.head())
df = extract_features(df)

#we export to csv to not do the extract features every time
df.to_csv("nlp_features_train.csv", index=False)


#########LOAD AGAIN THE EXTRACTED FEATURE MATRIX AND PERFORM 5 FOLD CROSS VAL ON SVC




df = pd.read_csv("nlp_features_train.csv")
df = df.drop([ 'Question1','Question2'], axis = 1)
df = df[df['IsDuplicate'].notna()]
print(df.isnull().values.any())

labels = np.array(df['IsDuplicate'])
features= df.drop('IsDuplicate', axis = 1)

feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)
print(features)
min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)


clf =  SVC(kernel='linear')
scoring = {'accuracy': 'accuracy',
           'recall': 'recall',
           'precision': 'precision',
           'f1':'f1'}
#5-fold cross validation
cross_val_scores = cross_validate(clf, features, labels, cv=5, scoring=scoring)
print(cross_val_scores)


######################NOW FIT THE MODEL
clf.fit(features, labels)


##### NOW READ THE TEST FILE FOR PREDICTING
test=pd.read_csv(r"test_without_labels.csv")
print('read')
##### EXTRACT FEATURES OF THE TEST CSV
print("Extracting features for testing:")
test = test.fillna('')

test = extract_features(test)
test.to_csv("nlp_features_test.csv", index=False)
print('all good')
test = test.drop([ 'Question1','Question2'], axis = 1)
features= test

feature_list = list(features.columns)
print(test)


# Convert to numpy array
features = np.array(features)
print(features)
features = min_max_scaler.transform(test)
##### PREDICT AND SAVE
predictions=clf.predict(features)
output = pd.DataFrame(data={"Id": test["Id"], "Predicted": predictions})
output.to_csv('q2b_method2.csv', index=False, quoting=3)
