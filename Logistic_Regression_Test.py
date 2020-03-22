import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class LogisticRegressionClass(object):

    def __init__(self):
        pass

    def main(self):
        colNames = ['Sample_code_number', 'Clump_Thickness', 'Uniformity_of_Cell_Size', 'Uniformity_of_Cell_Shape',
                    'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
                    'Normal_Nucleoli', 'Mitoses', 'Class']
        data = pd.read_csv('breast-cancer-wisconsin.data', names=colNames)

        data = data.replace({'Class': {2: 0, 4: 1}})

        # Replacing the missing values with 1
        data = data.replace({'?': 1})

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

        logreg = LogisticRegression(random_state=0)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(
            logreg.score(X_test, np.ravel(y_test, order='C'))))

        confusion_matrix_out = confusion_matrix(y_test, np.ravel(y_pred, order='C'))
        print("Confusion Matrix: \n {}".format(confusion_matrix_out))
        print("Accuracy: {}".format(round((confusion_matrix_out[0][0] + confusion_matrix_out[1][1]) * 100.0 / (
                sum(confusion_matrix_out[0]) + sum(confusion_matrix_out[1])), 2)))
        print(
            "Malignant recall: {}".format(round(confusion_matrix_out[1][1] * 100.0 / sum(confusion_matrix_out[1]), 2)))


if __name__ == "__main__":
    obj = LogisticRegressionClass()
    obj.main()
