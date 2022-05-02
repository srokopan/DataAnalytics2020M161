import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


#Read csv using pandas
train=pd.read_csv("train.csv",engine="python",error_bad_lines=False)
#vectorize
vectorizer = TfidfVectorizer()
#grab title + content
X=vectorizer.fit_transform(train['Title']+train['Content'])
#svd
svd = TruncatedSVD()
svd.fit(X)
X=svd.transform(X)

#grab labels
y=train['Label']

# Random Forest Classifier
clf=RandomForestClassifier()


scoring = {'accuracy': 'accuracy',
           'recall_weighted': 'recall_weighted',
           'precision_weighted': 'precision_weighted',
           'f1_weighted':'f1_weighted'}

#5-fold
cross_val_scores = cross_validate(clf, X, y, cv=KFold(5), scoring=scoring)
print("Average test Accuracy of 5-fold cross validation is ", np.mean(cross_val_scores['test_accuracy']))
print("Average test Recall_Weighted of 5-fold cross validation is ", np.mean(cross_val_scores['test_recall_weighted']))
print("Average test Precision_Weighted of 5-fold cross validation is ", np.mean(cross_val_scores['test_precision_weighted']))
print("Average test F1 of 5-fold cross validation is ", np.mean(cross_val_scores['test_f1_weighted']))



# train
clf.fit(X,y)
test=pd.read_csv("test_without_labels.csv")
X_test=vectorizer.transform(test['Title']+test['Content'])

X_test=svd.transform(X_test)
# predict test
predictions=clf.predict(X_test)
output = pd.DataFrame(data={"Id": test["Id"], "Predicted": predictions})
output.to_csv('randomforest_svd.csv', index=False, quoting=3)
print('done')