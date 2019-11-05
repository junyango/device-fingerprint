import os
import pickle
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initializing variables to find best model
best_model = ""
best_score = 0

# Switch off all warnings
warnings.filterwarnings("ignore")

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
    print("\n" + "-" * 50)

    return model.score(x_test, y_test)


###############################################################################################
# Reading training dataset
###############################################################################################
root_dir = os.path.abspath("./")
for file in os.listdir(root_dir):
    if "features.csv" in file:
        feature_file = file

# Train set
full_feature_file_path = os.path.join(root_dir, feature_file)
gyro_data = pd.read_csv(full_feature_file_path, index_col=0)
labels = list(gyro_data['phone'])
gyro_data = gyro_data.drop("phone", axis=1)
label_num = []
for label in labels:
    label_num.append(phone_models[label])

X_train, X_test, y_train, y_test = train_test_split(gyro_data, label_num, test_size=0.3, random_state=42)

# ##############################################################################################
# # Evaluating Random Forest
# # Feature importance shows decision tree feature decision importance
# ##############################################################################################
#
# # RandomForest
# model_rfc = RandomForestClassifier(n_estimators=1000)
# model_rfc.fit(X_train, y_train)
#
# # Feature importance
# import numpy as np
# import matplotlib.pyplot as plt
#
# importances = model_rfc.feature_importances_
# indices = np.argsort(importances)
# plt.title("Feature importances")
# plt.barh(range(len(indices)), importances[indices])
# plt.yticks(range(len(indices)), [gyro_data.columns[i] for i in indices])
# plt.xlabel("Relative importance")
# plt.show()
# score = evaluate(model_rfc, X_train, X_test, y_train, y_test)
# best_score = score
# best_model = "random_forest"

################################################################################################
# Evaluating SVM kernels and tuning the hyperparameters C and gamma
# C is a regularization parameter that controls the trade off between the achieving a low training error
# Intuitively, the gamma parameter defines how far the influence of a single training example reaches
################################################################################################
from sklearn import svm
from sklearn.model_selection import GridSearchCV

# Computing for various kernels
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma': gammas}

################################################################################################
# Linear kernel
################################################################################################
print("Evaluating hyperparameters for linear kernel...")
grid_search = GridSearchCV(svm.SVC(kernel='linear'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
model_svm_linear = svm.SVC(kernel="linear", gamma=grid_search.best_params_["gamma"], C=grid_search.best_params_["C"])
score = evaluate(model_svm_linear, X_train, X_test, y_train, y_test)

# Checking and updating best score
if best_score < score:
    best_score = score
    best_model = "svm_linear"

################################################################################################
# RBF Kernel
################################################################################################
print("Evaluating hyperparameters for RBF kernel...")
grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
model_svm_rbf = svm.SVC(kernel="rbf", gamma=grid_search.best_params_["gamma"], C=grid_search.best_params_["C"])
score = evaluate(model_svm_rbf, X_train, X_test, y_train, y_test)

if best_score < score:
    best_score = score
    best_model = "svm_rbf"

################################################################################################
# Poly kernel
################################################################################################
print("Evaluating hyperparameters for Poly kernel...")
grid_search = GridSearchCV(svm.SVC(kernel='poly'), param_grid, cv=5)
grid_search.fit(X_train, y_train)
model_svm_poly = svm.SVC(kernel="poly", gamma=grid_search.best_params_["gamma"], C=grid_search.best_params_["C"])
score = evaluate(model_svm_poly, X_train, X_test, y_train, y_test)

if best_score < score:
    best_score = score
    best_model = "svm_rbf"

###############################################################################################
# Saving the best model
###############################################################################################
model_name = 'final_model.sav'
if best_model == "svm_linear":
    pickle.dump(model_svm_linear, open(model_name, 'wb'))
elif best_model == "svm_rbf":
    pickle.dump(model_svm_rbf, open(model_name, 'wb'))
elif best_model == "svm_poly":
    pickle.dump(model_svm_poly, open(model_name, 'wb'))
# elif best_model == "random_forest":
#     pickle.dump(model_rfc, open(model_name, 'wb'))



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
