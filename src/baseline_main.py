#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
import time
from utils import get_dataset
from make_data import *
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar,SimpleMLP,SimpleCNN


if __name__ == '__main__':
    args = args_parser()
    # if args.gpu:
    #     torch.cuda.set_device(args.gpu)
    # device = 'cuda' if args.gpu else 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load datasets
    train_df, test_df = load_data(args)
    print(train_df.shape)
    save_data(train_df, test_df, args)
    X_train,y_train,X_test,y_test=preprocess_data(train_df,test_df)
    
    X_train_array = X_train.values.astype(np.float32)
    X_train_tensor = torch.tensor(X_train_array, dtype=torch.float32) 
     # Convert y_train to numpy array and then to torch tensor
    y_train_array = y_train.values.astype(np.int64)
    y_train_tensor = torch.tensor(y_train_array, dtype=torch.long)

    X_test_array = X_test.values.astype(np.float32)
    X_test_tensor = torch.tensor(X_test_array, dtype=torch.float32)
    # Convert y_test to numpy array and then to torch tensor
    y_test_array = y_test.values.astype(np.int64)
    y_test_tensor = torch.tensor(y_test_array, dtype=torch.long)

    train_dataset=TensorDataset(X_train_tensor.unsqueeze(1), y_train_tensor)
    test_dataset=TensorDataset(X_test_tensor.unsqueeze(1), y_test_tensor)
    # train_dataset, test_dataset,user_groups = get_dataset2(X_train, y_train, X_test, y_test,args)
    # train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
        elif args.dataset == 'har':
            global_model =SimpleCNN(1,6)
    elif args.model == 'mlp':
        # # Multi-layer preceptron
        # img_size = train_dataset[0][0].shape
        # len_in = 1
        # for x in img_size:
        #     len_in *= x
        #     global_model = MLP(dim_in=len_in, dim_hidden=64,
        #                        dim_out=args.num_classes)
        global_model=SimpleMLP(561,200,6)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    start_time = time.time()
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    epoch_loss = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)

    total_run_time_seconds = time.time() - start_time
    total_run_time_minutes = total_run_time_seconds / 60
    total_run_time_hours = total_run_time_seconds / 3600
    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    print('\n Total Run Time: {0:0.4f} seconds'.format(total_run_time_seconds))
    print(' Total Run Time: {0:0.4f} minutes'.format(total_run_time_minutes))
    print(' Total Run Time: {0:0.4f} hours'.format(total_run_time_hours))

    # Plot loss
    save_dir = './save'
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig(os.path.join(save_dir, 'nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs)))


    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    print('Test on', len(test_dataset), 'samples')
    print("Test Accuracy: {:.2f}%".format(100*test_acc))
