import os
import pandas as pd
import features_extraction

# pre-processing
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Constant to declare for the parameters for the sliding window
WINDOW_SIZE = 512
STEP = 128
order = 3
f_sample = 100
f_cutoff = 30
nyquist_rate = f_cutoff / (f_sample / 2)


# Data path should arrive in the format of <phone>/<vibrated>/<frequency>/<.csv> file
def create_windows(data_path):
    columns = ["x", "y", "z"]
    df = pd.read_csv(data_path, usecols=[1, 2, 3], header=None)
    # Assigning columns to the data set
    df.columns = columns
    data_readings = df.values
    # # `noise reduction
    for axis in range(0, 3):
        # order 3 lowpass butterworth filter
        b, a = signal.butter(order, nyquist_rate)
        data_readings[:, axis] = signal.filtfilt(b, a, data_readings[:, axis])

    ###############################################################################################
    # Trying different scaling functions
    ###############################################################################################
    # Min Max Scaler
    # min_max_scaler = preprocessing.MinMaxScaler()
    # data_readings = min_max_scaler.fit_transform(data_readings)

    # Standard Scaler
    scaler = StandardScaler()
    data_readings = scaler.fit_transform(data_readings)
    num_rows = len(data_readings)
    all_window = []

    for start in range(0, num_rows - WINDOW_SIZE, STEP):
        window = data_readings[start:start + WINDOW_SIZE]
        all_window.append(window)

    return all_window


# Getting labels for phone data
phone_list = []
data_dir = os.path.abspath("./100hz_data")
for phone in os.listdir(data_dir):
    phone_list.append(phone)

columns = ['min_x', 'min_y', 'min_z', 'min_rss', 'max_x', 'max_y', 'max_z', 'max_rss', 'std_x', 'std_y', 'std_z',
           'std_rss', 'mean_x', 'mean_y', 'mean_z', 'mean_rss', 'variance_x', 'variance_y', 'variance_z',
           'variance_rss', 'kurtosis_x', 'kurtosis_y', 'kurtosis_z', 'kurtosis_rss',
           'skew_x', 'skew_y', 'skew_z', 'skew_rss', 'phone']

main_df = pd.DataFrame(columns=columns)

# Setting up to traverse through CSV files to extract features
for folder in os.listdir(data_dir):
    data_segments = []
    vibrated_folder = os.path.abspath(os.path.join(data_dir, folder))
    list_of_csv = os.listdir(vibrated_folder)
    for csv in list_of_csv:
        csv_path = os.path.join(vibrated_folder, csv)

        # Making the entire CSV file return a list of list of segments
        data_segments = create_windows(csv_path)

        # For each segment, i extract the features
        for i in data_segments:
            features_csv = []
            features_csv = features_extraction.extract_features(i)
            label = str(folder)
            features_csv.append(label)
            main_df = main_df.append(pd.Series(features_csv, index=main_df.columns), ignore_index=True)

# Export to CSV for machine learning
main_df.to_csv("./features.csv")

# ##################################################################
# # For graph plotting and visualization purposes
# ##################################################################
#
# data = "D:\\Y4S1\\CS4276\\device-fingerprint\\data\\black_huawei\\gyro_100hz_14102019_143558.csv"
# # data = "D:\\Y4S1\\CS4276\\device-fingerprint\\data\\htc_u11\\gyro_100hz_16102019_205643.csv"
# # # data = "D:\\Y4S1\\CS4276\\device-fingerprint\\data\\mi_max\\gyro_100hz_14102019_212145.csv"
# data1 = "D:\\Y4S1\\CS4276\\device-fingerprint\\data\\black_huawei\\gyro_100hz_14102019_143651.csv"
# #
# name1 ="black_huawei"
# name2 ="black_huawei"
# #
# #
# # # For Black huawei
# data_segments = create_segments(data)
# # data_windows = create_window(data)
# #
# # # for blue huawei
# data1_segments = create_segments(data1)
# # data1_windows = create_window(data1)
# #
# # for black huawei
# features_segment = []
# for i in data_segments:
#     features_segment.append(features_extraction.extract_features(i))
#
# # for blue huawei
# features_segment1 = []
# for i in data1_segments:
#     features_segment1.append(features_extraction.extract_features(i))
#
#
# for j in range(len(features_segment[0])):
#     list1 = []
#     list2 = []
#     for i in features_segment:
#         list1.append(i[j])
#
#     for i in features_segment1:
#         list2.append(i[j])
#
#     plt.plot(list1, label=name1)
#     plt.plot(list2, label=name2)
#     plt.xlabel("segments")
#     plt.ylabel("readings")
#     legend = plt.legend(loc='upper center', shadow=True, fontsize='large')
#     if not os.path.exists("./" + name1 + "_" + name2):
#         os.mkdir("./" + name1 + "_" + name2)
#     plt.savefig("./" + name1 + "_" + name2 + "/" + str(j) + name1 + "_" + name2 + '.png')
#     plt.clf()
#
