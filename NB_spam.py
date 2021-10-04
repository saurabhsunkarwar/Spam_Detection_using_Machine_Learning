
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

df = pd.read_csv('D:/side project/sms spam classifier/spam.csv', encoding="latin-1")
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

df['label'] = df['v1'].map({'ham': 0, 'spam': 1})

X = df['v2']
y = df['label']

cv = CountVectorizer()

# Fit the Data
X = cv.fit_transform(X) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

'''
After training the model, it is desirable to have a way to persist the model for future use
 without having to retrain. To achieve this, we add the following lines to save our model
 as a .pkl file for the later use.'''

#from sklearn.externals import joblib
import joblib
joblib.dump(clf, 'NB_spam_model.pkl')

#And we can load and use saved model later like so:
NB_spam_model = open('NB_spam_model.pkl','rb')
clf = joblib.load(NB_spam_model)