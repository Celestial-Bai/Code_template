import numpy as np
import torch
from basicsetting import *
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from copy import deepcopy
import pandas as pd

class EarlyStopping:
    """Early stops the valing if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint_nn_v1.pt')
        self.val_loss_min = val_loss

class NN_encoder(torch.nn.Module):
    def __init__(self):
        super(NN_encoder, self).__init__()
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(561, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(256, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(64, 32),
        )
        #self.out = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.dense(x)
        return x

class NN_decoder(torch.nn.Module):
    def __init__(self):
        super(NN_decoder, self).__init__()
        self.dense = torch.nn.Sequential(
            torch.nn.BatchNorm1d(32),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(32, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(64, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(256, 561),
        )
        #self.out = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.dense(x)
        return x


class MAE(torch.nn.Module):
    def __init__(self):
        super(MAE, self).__init__()
        self.encoder = NN_encoder()
        self.decoder = NN_decoder()
        #self.out = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class trainset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __getitem__(self, index):
        Xvariables = self.X[index]
        labels = self.y[index]
        return Xvariables,labels
    def __len__(self):
        return len(self.X)

def mask_encoding(trainset, mask_ratio=0.75):
    labels = trainset.clone()
    probability_matrix = torch.full(labels.shape, mask_ratio)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    inputs = trainset.clone()
    masks = deepcopy(masked_indices)
    for i, masked_index in enumerate(masks):
        mask_centers = torch.where(masked_index == 1)[0]
        inputs[i][mask_centers] = 100
    return inputs, labels


if __name__ == '__main__':
    tr_batchsize = 128
    val_batchsize = 128
    train_data = pd.read_csv('X_train.csv')
    train_data = np.array(train_data, dtype='float32')
    X_train = torch.tensor(train_data)
    model = MAE()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    loss_func = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(),lr=1e-2)
    print("start to train")
    train_loss_value=[]
    train_acc_value=[]
    for epoch in range(100):
        inputs, labels = mask_encoding(X_train)
        train_data = trainset(X=inputs, y=labels)
        train_loader = DataLoader(train_data, batch_size=tr_batchsize, shuffle=True, num_workers=0)
        running_loss = 0.0
        valrun_loss = 0.0
        sum_correct = 0
        sum_total = 0
        t = len(train_loader.dataset)
        model.train()
        with tqdm(total=100) as pbar:
            for i, (x, y) in enumerate(train_loader):
                X_seq = x
                X_seq = X_seq.to(device)
                Label = Variable(y)
                Label = Label.to(device)
                opt.zero_grad()
                out = model(X_seq)
                loss = loss_func(out, Label)
                loss.backward()
                opt.step()
                # print statistics
                running_loss += loss.item()
                pbar.update(100 * tr_batchsize / t)
            pbar.close()
        print("epochs={}, mean loss={}"
            .format(epoch + 1, running_loss * tr_batchsize / t))
        train_loss_value.append(running_loss * tr_batchsize / t)
    torch.save(model.module.encoder.state_dict(), 'MAE_full_att10.pt')

