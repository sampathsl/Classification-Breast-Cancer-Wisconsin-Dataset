import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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

X = data_final.loc[:, data_final.columns != 'Class']
y = data_final.loc[:, data_final.columns == 'Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.349, random_state=1)

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(X_train, y_train.values.ravel())

y_pred = model.predict(X_test)

from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("\nConfusion Matrix: \n{}\n".format(confusion_matrix(y_test, y_pred)))
print("\nClassification Report: \n{}\n".format(classification_report(y_test, y_pred)))


