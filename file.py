import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
dta =sm.datasets.fair.load_pandas().data
#add &quot;affair&quot; column: 1 represents having affairs, 0 represents not
dta['affair'] = (dta.affairs>0).astype(int)
y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + religious + educ + C(occupation) + C(occupation_husb)',dta, return_type="dataframe")
X = X.rename(columns =
{'C(occupation)[T.2.0]':'occ_2',
'C(occupation)[T.3.0]':'occ_3',
'C(occupation)[T.4.0]':'occ_4',
'C(occupation)[T.5.0]':'occ_5',
'C(occupation)[T.6.0]':'occ_6',
'C(occupation_husb)[T.2.0]':'occ_husb_2',
'C(occupation_husb)[T.3.0]':'occ_husb_3',

'C(occupation_husb)[T.4.0]':'occ_husb_4',
'C(occupation_husb)[T.5.0]':'occ_husb_5',
'C(occupation_husb)[T.6.0]':'occ_husb_6'})
y = np.ravel(y)
X.drop('Intercept',axis=1,inplace=True)
n = 1
plt.figure(figsize=(20, 25))

for i in X:
    if n <= 17:
        plt.subplot(5, 4, n)
        sns.distplot(X[i])
        n += 1

n = 1
plt.figure(figsize=(20, 25))
for i in X:
    if n <= 17:
        plt.subplot(5,4,n)
        sns.countplot(x= X[i])
        n+=1



scaler = StandardScaler()
scaled= scaler.fit_transform(X)
print(scaled)

X_train, X_test, y_train, y_test = train_test_split(scaled,y,test_size=0.25,random_state=250)

classifier = LogisticRegression()
classifier.fit(X_train,y_train)

import pickle

pickle.dump(classifier,open('logisticModel.sav','wb'))
pickle.dump(scaler,open('scaler.sav','wb'))

from sklearn.metrics import confusion_matrix
y_pre = classifier.predict(X_test)
con_mat = confusion_matrix(y_test,y_pre)
sns.heatmap(con_mat ,annot=True)
plt.show()
print(con_mat)

plt.figure(figsize=(20,25))
sns.boxplot(data=X)
plt.show()




