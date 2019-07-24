import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('C:/Users/DELL/PycharmProjects/AI&ML/venv/instagram spam or fake.csv')
x = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].values
y = df.iloc[:, 11].values

sb.countplot(df['fake'], label='count')

fake_false = df[df['fake'] == 0]

sb.heatmap(df.corr(), annot=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
'''logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
logreg.score(x_test, y_test)*100'''

#accuracy = 90.27777777777779


knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
knn.score(x_test, y_test)*100

#accuracy = 88.19444444444444

dt = DecisionTreeClassifier(random_state=1)
dt.fit(x_train, y_train)
dt.score(x_test, y_test)*100

#accuracy = 86.80555555555556

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
rf.score(x_test, y_test)*100

#accuracy = 90.97222222222221



y_predict_train = rf.predict(x_train)


cm = confusion_matrix(y_train, y_predict_train)
sb.heatmap(cm, annot=True)

y_predict_test = rf.predict(x_test)

cm = confusion_matrix(y_test, y_predict_test)
sb.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict_test))

res = rf.predict(x_train[[0]])
print(res)

# y_train[0]

'''res = logreg.predict(x_train[[0]])
print(res)'''


for i in range(200):
    r=dt.predict(x_train[[i]])
    print(r, y_train[i])

# test cases

info = [[1, 0.25, 3, 0, 0, 70, 0, 0, 1, 107, 107]]
rf.predict(info)

i = rf.predict([[0, 0, 1, 0, 0, 0, 0, 1, 0, 399, 7496]])
print(i)

ad = [[1, 0.0, 3, 0, 0, 150, 1, 1, 1, 319, 205]]
g = rf.predict(ad)
print(g)

s = [[1, 0, 3, 0, 0, 78, 0, 0, 101, 279, 122]]
rf.predict(s)





