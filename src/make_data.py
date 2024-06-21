import pandas as pd
import argparse
from sklearn.utils import shuffle
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid
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


def preprocess_data(train_df, test_df):
    # Subtract 1 from 'activity' column in both train and test DataFrames
    train_df['activity'] -= 1
    test_df['activity'] -= 1

    # Split data into features (X) and labels (y)
    y_train = train_df['activity']
    X_train = train_df.drop(['activity', 'activity_name', 'subject_id'], axis=1)
    y_test = test_df['activity']
    X_test = test_df.drop(['activity', 'activity_name', 'subject_id'], axis=1)

    return X_train, y_train, X_test, y_test


def create_datasets(X_train, y_train, X_test, y_test):
    # Convert X_train to numpy array and then to torch tensor
    X_train_array = X_train.values.astype(np.float32)
    X_train_tensor = torch.tensor(X_train_array, dtype=torch.float32)
    
    # Convert y_train to numpy array and then to torch tensor
    y_train_array = y_train.values.astype(np.int64)
    y_train_tensor = torch.tensor(y_train_array, dtype=torch.long)
    
    # Create TensorDataset for training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    
    # Convert X_test to numpy array and then to torch tensor
    X_test_array = X_test.values.astype(np.float32)
    X_test_tensor = torch.tensor(X_test_array, dtype=torch.float32)
    
    # Convert y_test to numpy array and then to torch tensor
    y_test_array = y_test.values.astype(np.int64)
    y_test_tensor = torch.tensor(y_test_array, dtype=torch.long)
    
    # Create TensorDataset for testing data
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
    else:
        if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
        else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)
    
    return train_dataset, test_dataset,user_groups

  
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
    parser.add_argument("num_users", type=int, help="Num of users")

    return parser

if __name__ == "__main__":
    parser = args_parser()
    args = parser.parse_args()
    train_df, test_df = load_data(args.data_dir)
    save_data(train_df, test_df, args.output_dir)
    X_train,y_train,X_test,y_test=preprocess_data(train_df,test_df)
    train_dataset, test_dataset,user_groups = create_datasets(X_train, y_train, X_test, y_test)
    print(train_dataset[0])
    print(test_dataset[0])
    print(user_groups[0]

