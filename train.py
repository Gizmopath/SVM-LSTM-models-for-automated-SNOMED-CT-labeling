import utils
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("clean_dataset.csv")
df = df.fillna('')

X = df["Sentences"]
y = df["Labels"]

max_df = 
min_df = 
kernel = "linear"
c_val = 0.5

vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
X = vectorizer.fit_transform(X)
utils.save_vocabulary(vectorizer.vocabulary_)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

clf = svm.SVC(kernel="linear", C=c_val, decision_function_shape='ovr')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1_score_micro = f1_score(y_test, y_pred, average="micro")
f1_score_macro = f1_score(y_test, y_pred, average="macro")

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("F1 Score (micro): {:.2f}%".format(f1_score_micro * 100))
print("F1 Score (macro): {:.2f}%".format(f1_score_macro * 100))
