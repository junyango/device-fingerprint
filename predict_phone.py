import os
import numpy as np
import pandas as pd
import warnings
from collections import Counter
import pickle

import features_extraction
import features_generation

phone_models_decode = {
    1: "black_huawei",
    2: "galaxy_note5",
    3: "htc_u11",
    4: "pixel_2",
    5: "blue_huawei",
    6: "galaxy_tabs",
    7: "mi_max"
}

warnings.filterwarnings("ignore")

######################################################################################################
# TEST DATA
# Extract features from test data
######################################################################################################
test_columns = ['min_x', 'min_y', 'min_z', 'min_rss', 'max_x', 'max_y', 'max_z', 'max_rss', 'std_x', 'std_y', 'std_z',
                'std_rss', 'mean_x', 'mean_y', 'mean_z', 'mean_rss', 'variance_x', 'variance_y', 'variance_z',
                'variance_rss', 'kurtosis_x', 'kurtosis_y', 'kurtosis_z', 'kurtosis_rss',
                'skew_x', 'skew_y', 'skew_z', 'skew_rss']
test_df = pd.DataFrame(columns=test_columns)
test_dir = os.path.abspath("./test")

# Setting up to traverse through CSV files to extract features
for csv in os.listdir(test_dir):
    data_segments = []
    csv_path = os.path.join(test_dir, csv)

    # Making the entire CSV file return a list of list of segments
    data_segments = features_generation.create_windows(csv_path)

    # For each segment, i extract the features
    for i in data_segments:
        features_csv = []
        features_csv = features_extraction.extract_features(i)
        test_df = test_df.append(pd.Series(features_csv, index=test_df.columns), ignore_index=True)

# Export test data to CSV
# Unnecessary actually
test_df.to_csv("./test_data.csv")


def prediction(model, x_test):
    results = list(model.predict(X_test))
    data = Counter(results)
    most_common = data.most_common(1)[0][0]
    count = data[most_common]
    percentage_accuracy = (count/len(results)) * 100
    print("Confidence level: " + str(percentage_accuracy))
    return phone_models_decode[most_common]


# Initializing root path
root_dir = os.path.abspath("./")

################################################################################################
# Loading model
################################################################################################
loaded_model = pickle.load(open(os.path.join(root_dir, "final_model.sav"), 'rb'))

###############################################################################################
# Prediction
###############################################################################################
# Read dataset to pandas dataframe
for file in os.listdir(root_dir):
    if "test_data.csv" in file:
        test_file = file

full_feature_test_path = os.path.join(root_dir, test_file)
test_data = pd.read_csv(full_feature_test_path, index_col=0)

X_test = np.asarray(test_data)
result = prediction(loaded_model, test_data)
print("The phone detected is: " + result)
