from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
import pandas as p

data = p.read_csv(r"classifier/data.csv", sep=";")
ih = data["Ill_Healthy"]
hgb = data["Hgb"]
hct = data["Hct"]
rbc = data["Rbc"]
mcv = data["MCV"]
arg, ans = [], []
#list_data = [float(value) for value in input().split()]
# print(list_data)

for i in range(len(data)):
    arg.append([hgb[i], hct[i], rbc[i], mcv[i]])
    ans.append(ih[i])

arg_train = arg
ans_train = ans
clf = LogisticRegression(random_state=0, max_iter=10000, class_weight={1:4, 0:1})
clf.fit(arg_train, ans_train)
# print(arg_train)

def get_prediction(values):
    return clf.predict([values])[0]


