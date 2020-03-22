import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Loading data
colNames = ['Sample_code_number', 'Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape',
            'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
            'Normal_Nucleoli', 'Mitoses', 'Class']
data = pd.read_csv("breast-cancer-wisconsin.data", names=colNames)
data.head()

# Data pre processing
data = data.replace({'Class': {2: "Benign", 4: "Malignant"}})

# Remove data wich has missing values
data = data[data.Bare_Nuclei != "?"]

total_samples = data['Sample_code_number'].count()
print("Number of columns\t: {}".format(total_samples))
cat_vars = ['Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape',
            'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
            'Normal_Nucleoli', 'Mitoses', 'Class']
data_final = data[cat_vars]
data_final.head()

X = data_final.loc[:, data_final.columns != 'Class']
y = data_final.loc[:, data_final.columns == 'Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.349, random_state=1)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y.head()

# tree.plot_tree(clf.fit(iris.data, iris.target))

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
    clf.score(X_test, np.ravel(y_test, order='C'))))
y_pred = clf.predict(X_test)
confusion_matrix_out = confusion_matrix(y_test, np.ravel(y_pred, order='C'))
print("Confusion Matrix: \n {}".format(confusion_matrix_out))
print("Accuracy: {}".format(round((confusion_matrix_out[0][0] + confusion_matrix_out[1][1]) * 100.0 / (
        sum(confusion_matrix_out[0]) + sum(confusion_matrix_out[1])), 2)))
print("Malignant recall: {}".format(round(confusion_matrix_out[1][1] * 100.0 / sum(confusion_matrix_out[1]), 2)))

from chefboost import Chefboost as chef
import pandas as pd
from sklearn.model_selection import train_test_split

# Loading data
colnames = ['Sample_code_number', 'Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape',
            'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
            'Normal_Nucleoli', 'Mitoses', 'Decision']
data = pd.read_csv("breast-cancer-wisconsin.data", names=colnames)
data.head()


# Data pre processing

data = data.replace({'Decision': {2: 0, 4: 1}})

total_samples = data['Sample_code_number'].count()
print("Number of columns\t: {}".format(total_samples))

# Remove the code number from data
final_data = data.drop(['Sample_code_number'], axis=1)
final_data.head()


# Split train and test set
df = final_data.copy()
X = df.loc[:, df.columns != 'Decision']
y = df.loc[:, df.columns == 'Decision']
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.349, random_state=1)
training_set = pd.concat([X_train, y_train], axis=1)
print("Training set size: {}".format(X_train.Mitoses.count()))
print("Testing set size: {}".format(X_test.Mitoses.count()))


X_train.head()
y_train.head()

# Training
# config = {'algorithm': 'ID3'}
config = {'algorithm': 'C4.5'}
model = chef.fit(training_set, config)

X_test.Clump_Thickness.count()
y_test.head()


# Calculate Accuracy
_true = 0
_false = 0
accuracy = {"Benign": {"Malignant": 0, "Benign": 0}, "Malignant": {"Malignant": 0, "Benign": 0}}
for i in range(X_test.Clump_Thickness.count()):
    prediction = chef.predict(model, X_test.iloc[i])
    if prediction != None and round(prediction) == y_test.iloc[i].Decision:
        _true += 1
        if y_test.iloc[i].Decision == 0:
            accuracy["Benign"]["Benign"] += 1
        else:
            accuracy["Malignant"]["Malignant"] += 1
    else:
        _false += 1
        if y_test.iloc[i].Decision == 0:
            accuracy["Benign"]["Malignant"] += 1
        else:
            accuracy["Malignant"]["Benign"] += 1
print(accuracy)
print("\nTotal Accuracy: {:0.2f}".format(_true * 100 / (_true + _false)))
print(
    "\nConsfusion Matrix:\nGround Truth | Prediction- \tBenign Malignant\nBenign\t\t\t\t{}\t{}\nMalignant\t\t\t{}\t{}".format(
        accuracy["Benign"]["Benign"], accuracy["Benign"]["Malignant"], accuracy["Malignant"]["Benign"],
        accuracy["Malignant"]["Malignant"]))
print("\nMalignant Recall: {:0.2f}".format(
    accuracy["Malignant"]["Malignant"] * 100 / (accuracy["Malignant"]["Benign"] + accuracy["Malignant"]["Malignant"])))
print("\n")

pred_loc = 1
print(df.iloc[pred_loc])
prediction = chef.predict(model, df.iloc[pred_loc])
print("\nPrediction:", prediction)
