import pandas as pd
import numpy as np
import argparse
from sklearn.utils import shuffle
import torch

from torch.utils.data import TensorDataset, DataLoader

def load_data(data_dir):
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

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs =919,8
    #200, 300
    idx_shard = [i for i in range(num_shards)]
    print(idx_shard)
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    print(len(idxs))
    labels = dataset.tensors[1].numpy()
    print(len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def get_dataset2(X_train, y_train, X_test, y_test,args):
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

def args_parser2():
    parser = argparse.ArgumentParser(description="Load and Save UCI HAR Dataset")
    parser.add_argument("data_dir", type=str, help="Directory containing the UCI HAR Dataset")
    parser.add_argument("output_dir", type=str, help="Directory to save the processed CSV files")
    parser.add_argument("num_users", type=int, help="Num of users")
    parser.add_argument("--iid", action='store_true', help="Whether to use IID (True) or Non-IID (False) data distribution")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = args_parser2()
    train_df, test_df = load_data(args)
    print(train_df.shape)
    save_data(train_df, test_df, args)
    X_train,y_train,X_test,y_test=preprocess_data(train_df,test_df)
    train_dataset, test_dataset,user_groups = get_dataset2(X_train, y_train, X_test, y_test,args)
    print(f"Số lượng mẫu trong train_dataset: {len(train_dataset)}")
    print(train_dataset[0])
    print(train_dataset[0])
    print(test_dataset[0])
    # print(user_groups[0])
    trainloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    loader = DataLoader(list(zip(X_train,y_train)), shuffle=True, batch_size=16)
    for X_batch, y_batch in loader:
        print(X_batch, y_batch)
        print('------------------------------------')
        break
    for batch_idx, (data, target) in enumerate(trainloader):
        print(data, target)
        print(f'Batch {batch_idx + 1}:')
        print('Data shape:', data.shape)
        print('Target shape:', target.shape)
        break


