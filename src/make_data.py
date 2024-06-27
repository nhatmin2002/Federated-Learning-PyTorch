import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import torch
from sampling import mnist_iid,mnist_noniid
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import TensorDataset, DataLoader

def load_data(args):
    # Load feature names
    features = []
    with open(f"{args.data_dir}/UCI HAR Dataset/features.txt") as file:
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
    train_df = pd.read_csv(f"{args.data_dir}/UCI HAR Dataset/train/X_train.txt", delim_whitespace=True, names=features)
    train_df['subject_id'] = pd.read_csv(f"{args.data_dir}/UCI HAR Dataset/train/subject_train.txt", header=None).squeeze()
    train_df["activity"] = pd.read_csv(f"{args.data_dir}/UCI HAR Dataset/train/y_train.txt", header=None).squeeze()
    activity = pd.read_csv(f"{args.data_dir}/UCI HAR Dataset/train/y_train.txt", header=None).squeeze()
    label_name = activity.map({1: "WALKING", 2:"WALKING_UPSTAIRS", 3:"WALKING_DOWNSTAIRS", 4:"SITTING", 5:"STANDING", 6:"LYING"})
    train_df["activity_name"] = label_name
    
    # Load test data
    test_df = pd.read_csv(f"{args.data_dir}/UCI HAR Dataset/test/X_test.txt", delim_whitespace=True, names=features)
    test_df['subject_id'] = pd.read_csv(f"{args.data_dir}/UCI HAR Dataset/test/subject_test.txt", header=None).squeeze()
    test_df["activity"] = pd.read_csv(f"{args.data_dir}/UCI HAR Dataset/test/y_test.txt", header=None).squeeze()
    activity_test = pd.read_csv(f"{args.data_dir}/UCI HAR Dataset/test/y_test.txt", header=None).squeeze()
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

def PREPROCESS(X):
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    return X

def get_dataset2(X_train, y_train, X_test, y_test,args):
    sample_original = X_train.iloc[0]

    X_train_sc = PREPROCESS(X_train)
    X_test_sc = PREPROCESS(X_test)
    sample_scaled = X_train_sc[0]
    print('1',sample_original)
    print('2',sample_scaled)


    X_train_array = X_train_sc.astype(np.float32)
    X_train_tensor = torch.tensor(X_train_array, dtype=torch.float32)
    y_train_array = y_train.values.astype(np.int64)
    y_train_tensor = torch.tensor(y_train_array, dtype=torch.long)
    
    X_test_array = X_test_sc.astype(np.float32)
    X_test_tensor = torch.tensor(X_test_array, dtype=torch.float32)
    y_test_array = y_test.values.astype(np.int64)
    y_test_tensor = torch.tensor(y_test_array, dtype=torch.long)
    
    
    train_dataset=TensorDataset(X_train_tensor.unsqueeze(1), y_train_tensor)
    test_dataset=TensorDataset(X_test_tensor.unsqueeze(1), y_test_tensor)

    if args.iid:
        user_groups = mnist_iid(train_dataset, args.num_users)
    else:
        user_groups = mnist_noniid(train_dataset, args.num_users)
    
    return train_dataset, test_dataset,user_groups

  
def save_data(train_df, test_df,args):
    # Shuffle the dataframes
    train_df = shuffle(train_df)
    test_df = shuffle(test_df)
    
    # Save to CSV
    train_df.to_csv(f"{args.output_dir}/train.csv", index=False)
    test_df.to_csv(f"{args.output_dir}/test.csv", index=False)

# def args_parser2():
#     parser = argparse.ArgumentParser(description="Load and Save UCI HAR Dataset")
#     parser.add_argument("data_dir", type=str, help="Directory containing the UCI HAR Dataset")
#     parser.add_argument("output_dir", type=str, help="Directory to save the processed CSV files")
#     parser.add_argument("num_users", type=int, help="Num of users")
#     parser.add_argument("--iid", action='store_true', help="Whether to use IID (True) or Non-IID (False) data distribution")
#     args = parser.parse_args()

#     return args

# if __name__ == "__main__":
#     args = args_parser()
#     train_df, test_df = load_data(args)
#     print(train_df.shape)
#     save_data(train_df, test_df, args)
#     X_train,y_train,X_test,y_test=preprocess_data(train_df,test_df)
#     train_dataset, test_dataset,user_groups = get_dataset2(X_train, y_train, X_test, y_test,args)
#     # print(f"Số lượng mẫu trong train_dataset: {len(train_dataset)}")
#     # print(train_dataset[0])
#     # print(train_dataset[0])
#     # print(test_dataset[0])
#     # print(user_groups[0])
#     trainloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
#     X_train_array = X_train.values.astype(np.float32)
#     X_train_tensor = torch.tensor(X_train_array, dtype=torch.float32) 
#     # Convert y_train to numpy array and then to torch tensor
#     y_train_array = y_train.values.astype(np.int64)
#     y_train_tensor = torch.tensor(y_train_array, dtype=torch.long)
#     loader = DataLoader(list(zip(X_train_tensor,y_train_tensor)), shuffle=False, batch_size=16)
#     for X_batch, y_batch in loader:
#         print(X_batch, y_batch)
#         print('------------------------------------')
#         break
#     for batch_idx, (data, target) in enumerate(trainloader):
#         print(data, target)
#         print(f'Batch {batch_idx + 1}:')
#         print('Data shape:', data.shape)
#         print('Target shape:', target.shape)
#         break


