import pandas as pd
import argparse
from sklearn.utils import shuffle

def load_data(data_dir):
    # Load feature names
    features = []
    with open(f"{data_dir}/UCI HAR Dataset/features.txt") as file:
        for line in file:
            features.append(line.split()[1])
    
    # Renaming duplicate column names
    names = []
    count = {}
    for feature in features:
        if features.count(feature) > 1:
            names.append(feature)
    for name in names:
        count[name] = features.count(name)

    for i in range(len(features)):
        if features[i] in names:
            num = count[features[i]]
            count[features[i]] -= 1
            features[i] = str(features[i] + str(num))

    # Load train data
    train_df = pd.read_csv(f"{data_dir}/UCI HAR Dataset/train/X_train.txt", delim_whitespace=True, names=features)
    train_df['subject_id'] = pd.read_csv(f"{data_dir}/UCI HAR Dataset/train/subject_train.txt", header=None).squeeze()
    train_df["activity"] = pd.read_csv(f"{data_dir}/UCI HAR Dataset/train/y_train.txt", header=None).squeeze()
    activity = pd.read_csv(f"{data_dir}/UCI HAR Dataset/train/y_train.txt", header=None).squeeze()
    label_name = activity.map({1: "WALKING", 2:"WALKING_UPSTAIRS", 3:"WALKING_DOWNSTAIRS", 4:"SITTING", 5:"STANDING", 6:"LYING"})
    train_df["activity_name"] = label_name
    
    # Load test data
    test_df = pd.read_csv(f"{data_dir}/UCI HAR Dataset/test/X_test.txt", delim_whitespace=True, names=features)
    test_df['subject_id'] = pd.read_csv(f"{data_dir}/UCI HAR Dataset/test/subject_test.txt", header=None).squeeze()
    test_df["activity"] = pd.read_csv(f"{data_dir}/UCI HAR Dataset/test/y_test.txt", header=None).squeeze()
    activity_test = pd.read_csv(f"{data_dir}/UCI HAR Dataset/test/y_test.txt", header=None).squeeze()
    label_name_test = activity_test.map({1: "WALKING", 2:"WALKING_UPSTAIRS", 3:"WALKING_DOWNSTAIRS", 4:"SITTING", 5:"STANDING", 6:"LYING"})
    test_df["activity_name"] = label_name_test

    return train_df, test_df
  
def save_data(train_df, test_df, output_dir):
    # Shuffle the dataframes
    train_df = shuffle(train_df)
    test_df = shuffle(test_df)
    
    # Save to CSV
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

def args_parser():
    parser = argparse.ArgumentParser(description="Load and Save UCI HAR Dataset")
    parser.add_argument("data_dir", type=str, help="Directory containing the UCI HAR Dataset")
    parser.add_argument("output_dir", type=str, help="Directory to save the processed CSV files")
    return parser

if __name__ == "__main__":
    parser = args_parser()
    args = parser.parse_args()
    train_df, test_df = load_data(args.data_dir)
    save_data(train_df, test_df, args.output_dir)
