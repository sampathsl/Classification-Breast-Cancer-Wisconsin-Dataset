import warnings

import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class LogisticRegressionClass(object):

    def __init__(self):
        pass

    def main(self):
        # Loading data
        colnames = ['Sample_code_number', 'Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape',
                    'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
                    'Normal_Nucleoli', 'Mitoses', 'Class']
        data = pd.read_csv("breast-cancer-wisconsin.data", names=colnames)
        data.head()

        # Data pre processing
        data = data.replace({'Class': {2: "Benign", 4: "Malignant"}})

        # Replacing the missing values with 1
        data = data.replace({'?': 1})

        # Remove data wich has missing values
        # data = data[data.Bare_Nuclei != "?"]

        total_samples = data['Sample_code_number'].count()
        print("Number of columns\t: {}".format(total_samples))

        cat_vars = ['Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape',
                    'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
                    'Normal_Nucleoli', 'Mitoses', 'Class']
        data_final = data[cat_vars]

        X = data_final.loc[:, data_final.columns != 'Class']
        y = data_final.loc[:, data_final.columns == 'Class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.349, random_state=1)

        train_size = X_train.shape[0]
        test_size = X_test.shape[0]
        print("Train data set size\t: {} ({}%)".format(train_size, round(train_size * 100 / total_samples, 2)))
        print("Test data set size\t: {} ({}%)".format(test_size, round(test_size * 100 / total_samples, 2)))

        SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr',
            degree=3, gamma='scale', kernel='linear', max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        svclassifier = SVC(kernel='poly', degree=4)
        svclassifier.fit(X_train, y_train)

        y_pred = svclassifier.predict(X_test)
        print(confusion_matrix(y_test, y_pred))

        print(classification_report(y_test,y_pred))


if __name__ == "__main__":
    obj = LogisticRegressionClass()
    obj.main()
