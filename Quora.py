import numpy as np
import mglearn
import pandas as pd

# data loading
text_train = pd.read_csv("train_data.csv", header = None)
y_train = pd.read_csv("train_labels.csv", header = None)

text_train = text_train.drop(text_train.columns[0], axis=1)
y_train = y_train.drop(y_train.columns[0], axis=1)
y_train = y_train[1].values.tolist()
y_train = [str(i) for i in y_train]

X_test = pd.read_csv("test_data.csv")

# create list with questions merged for the test data

quest_1_test = []
quest_2_test = []

for i in X_test['question1']:
    quest_1_test.append(i)
    
for i in X_test['question2']:
    i = str(i)
    quest_2_test.append(i)

test_questions = []

for i, ii in zip(quest_1_test, quest_2_test):
    test_questions.append(str(i+' '+ii))

quest_1 = []
quest_2 = []

for i in text_train[1]:
    quest_1.append(i)
    
for i in text_train[2]:
    i = str(i)
    quest_2.append(i)

all_quest = quest_1 + quest_2 

# create one string per row (from both features)
questions = []

for i, ii in zip(quest_1, quest_2):
    questions.append(str(i+' '+ii))

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer((questions + test_questions), min_df=2)
X_all = tfidf.fit_transform((questions+test_questions))
X_train = X_all[0:239825]
X_test_tfidf = X_all[239825:]

from sklearn.ensemble import RandomForestClassifier
import datetime

print('{}: Fitting model...'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

forest = RandomForestClassifier(n_estimators=600, random_state=2, max_features = 'sqrt')
forest.fit(X_train, y_train)

test_predict = forest.predict(X_test_tfidf)

print('{}: finish.'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

output = {'id': X_test.iloc[:,0], 'is_duplicate': test_predict}
kaggle_output = pd.DataFrame(data=output)
kaggle_output.to_csv('final_output', index = False,sep = ',')
