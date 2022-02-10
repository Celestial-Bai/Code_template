from basicsetting import *
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
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
        torch.save(model.state_dict(), 'checkpoint_cnn_v1.pt')
        self.val_loss_min = val_loss

class GlobalMaxPooling1D(torch.nn.Module):

    def __init__(self, data_format='channels_last'):
        super(GlobalMaxPooling1D, self).__init__()
        self.data_format = data_format
        self.step_axis = 2 if self.data_format == 'channels_last' else 1

    def forward(self, input):
        return torch.max(input, axis=self.step_axis).values

class CNN(torch.nn.Module):
    def __init__(self, channels, n_filter, filter_size, n_dense):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(channels, n_filter, filter_size),
            torch.nn.ReLU(),
            GlobalMaxPooling1D(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(n_filter, n_dense),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(n_dense, 6)
        )
        # self.out = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
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

if __name__ == '__main__':
    n_filter = 512
    filter_size = 15
    n_dense = 512
    channels = 1
    tr_batchsize = 4
    val_batchsize = 4
    early_stopping = EarlyStopping(patience=10, verbose=True)

    train_data = pd.read_csv('X_train.csv')
    y = pd.read_csv('y_train.csv')
    y = y - 1
    X_train, X_val, y_train, y_val = train_test_split(train_data, y, test_size=0.33, random_state=0)
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.float32)
    X_train = torch.tensor(X_train)
    X_val = torch.tensor(X_val)
    y_train = torch.tensor(y_train)
    y_val = torch.tensor(y_val)
    train_data = trainset(X=X_train, y=y_train)
    train_loader = DataLoader(train_data, batch_size=tr_batchsize, shuffle=True, num_workers=3)
    val_data = trainset(X=X_val, y=y_val)
    val_loader = DataLoader(train_data, batch_size=val_batchsize, shuffle=True, num_workers=3)
    model = CNN(channels=channels, filter_size=filter_size, n_filter=n_filter, n_dense=n_dense)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(),lr=1e-2)
    print("start to train")
    train_loss_value=[]
    train_acc_value=[]
    for epoch in range(100):
        running_loss = 0.0
        valrun_loss = 0.0
        sum_correct = 0
        sum_total = 0
        t = len(train_loader.dataset)
        model.train()
        with tqdm(total=100) as pbar:
            for i, (x, y) in enumerate(train_loader):
                X_seq = x
                X_seq = X_seq.unsqueeze(1)
                X_seq = X_seq.to(device)
                Label = Variable(y)
                Label = Label.to(device)
                opt.zero_grad()
                out = model(X_seq)
                loss = loss_func(out.view(-1, 6), Label.view(-1).to(torch.long))
                loss.backward()
                opt.step()
                # print statistics
                running_loss += loss.item()
                pbar.update(100 * tr_batchsize / t)
            pbar.close()
        print("epochs={}, mean loss={}"
            .format(epoch + 1, running_loss * tr_batchsize / t))
        train_loss_value.append(running_loss * tr_batchsize / t)
        model.eval()
        t = len(val_loader.dataset)
        for i, (x, y) in enumerate(val_loader):
            X_seq = x
            X_seq = X_seq.unsqueeze(1)
            X_seq = X_seq.to(device)
            Label = Variable(y)
            Label = Label.to(device)
            val_output = model(X_seq)
            val_loss = loss_func(val_output.view(-1, 6), Label.view(-1).to(torch.long))
            valrun_loss += val_loss.item()
        early_stopping(valrun_loss * val_batchsize / t, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    torch.save(model.state_dict(), 'cnn_full_att1.pt')

