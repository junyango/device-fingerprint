import os

import numpy as np
import pandas as pd

# label phone models
phone_models = {
    "black_huawei": 1,
    "galaxy_note5": 2,
    "htc_u11": 3,
    "pixel_2": 4,
    "blue_huawei": 5,
    "galaxy_tabs": 6,
    "mi_max": 7
}


def evaluate(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    print("\n" + "-" * 15 + " " + type(model).__name__ + " " + "-" * 15)
    print("Accuracy on training set is : {}".format(model.score(x_train, y_train)))
    print("Accuracy on test set is : {}".format(model.score(x_test, y_test)))


# Read dataset to pandas dataframe
root_dir = os.path.abspath("./")
for file in os.listdir(root_dir):
    if "features.csv" in file:
        feature_file = file

# Train set
full_feature_file_path = os.path.join(root_dir, feature_file)
gyro_data = pd.read_csv(full_feature_file_path, index_col=0)
labels = list(gyro_data['phone'])
label_num = []
for label in labels:
    label_num.append(phone_models[label])

# X_train, X_test, y_train, y_test = train_test_split(gyro_data, labels, test_size=0.33, random_state=42)
X_train = np.asarray(gyro_data.drop("phone", axis=1))
print(X_train.shape)
y_train = np.asarray(label_num)

# Test set
# Read dataset to pandas dataframe
for file in os.listdir(root_dir):
    if "testtest.csv" in file:
        test_file = file

full_feature_test_path = os.path.join(root_dir, test_file)
test_data = pd.read_csv(full_feature_test_path, index_col=0)
test_labels = list(test_data['phone'])
test_label_num = []
for label in test_labels:
    test_label_num.append(phone_models[label])

X_test = np.asarray(test_data.drop("phone", axis=1))
print(X_test.shape)
y_test = np.asarray(test_label_num)


###############################################################################################
# Evaluating Random Forest
# Feature importance shows decision tree feature decision importance
###############################################################################################

# # RandomForest
# model = RandomForestClassifier(n_estimators=1000)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# from sklearn.metrics import confusion_matrix
# # Model Accuracy: how often is the classifier correct
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))

# # Feature importance
# import numpy as np
# importances = model.feature_importances_
# indices = np.argsort(importances)
# plt.title("Feature importances")
# plt.barh(range(len(indices)), importances[indices])
# plt.yticks(range(len(indices)), [gyro_data.columns[i] for i in indices])
# plt.xlabel("Relative importance")
# plt.show()
# evaluate(model, X_train, X_test, y_train, y_test)

###############################################################################################
# Evaluating SVM kernels and tuning the hyperparameters C and gamma
# C is a regularization parameter that controls the trade off between the achieving a low training error
# Intuitively, the gamma parameter defines how far the influence of a single training example reaches
###############################################################################################
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# Computing for various kernels
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma': gammas}
# Linear kernel
print("Evaluating hyperparameters for linear kernel...")
grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
model = svm.SVC(kernel="linear", gamma=grid_search.best_params_["gamma"], C=grid_search.best_params_["C"])
evaluate(model, X_train, X_test, y_train, y_test)

# RBF Kernel
print("Evaluating hyperparameters for RBF kernel...")
grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
model = svm.SVC(kernel="rbf", gamma=grid_search.best_params_["gamma"], C=grid_search.best_params_["C"])
evaluate(model, X_train, X_test, y_train, y_test)

# Poly kernel
print("Evaluating hyperparameters for Poly kernel...")
grid_search = GridSearchCV(svm.SVC(kernel='poly'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
model = svm.SVC(kernel="poly", gamma=grid_search.best_params_["gamma"], C=grid_search.best_params_["C"])
evaluate(model, X_train, X_test, y_train, y_test)


###############################################################################################
# Portion for KFold's validation
###############################################################################################

# scikit-learn k-fold cross-validation
# from sklearn.model_selection import KFold
#
# kf = KFold(n_splits=5, shuffle=True, random_state=None)
# # blank lists to store predicted values and actual values
# predicted_y = []
# expected_y = []
#
# for train_index, test_index in kf.split(X):
#     # print("Train index = ", train_index)
#     # print("Test index = ", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     # RandomForest
#     model = RandomForestClassifier(n_estimators=1000)
#     evaluate(model, X_train, X_test, y_train, y_test)
#
#     # store result from classification
#     predicted_y.extend(model.predict(X_test))
#
#     # store expected result for this specific fold
#     expected_y.extend(y_test)
#
#     # save and print accuracy
# accuracy = metrics.accuracy_score(expected_y, predicted_y)
# print("Accuracy: " + accuracy.__str__())
#
# # MaxVoting
# model1 = KNeighborsClassifier(n_neighbors=3)
# model2 = tree.DecisionTreeClassifier(random_state=1)
# model3 = sk.LogisticRegression(random_state=1, solver='liblinear')
# model = VotingClassifier(estimators=[('kn', model1), ('dt', model2), ('lr', model3)], voting='hard')
# evaluate(model, X_train, X_test, y_train, y_test)
#
# # Baggging Classifier
# model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
# evaluate(model, X_train, X_test, y_train, y_test)
#
# # SVM Classifier
# model = svm.SVC(kernel='rbf', C=1, random_state=2048)
# evaluate(model, X_train, X_test, y_train, y_test)
#
# # Naive Bayes Classifier
# model = GaussianNB()
# evaluate(model, X_train, X_test, y_train, y_test)
