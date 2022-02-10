from basicsetting import *
from nn_tr import *


if __name__ == '__main__':
    model = NN()
    sdict = torch.load('checkpoint_nn_full_v7.pt')
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(sdict)
    model = model.to(device)
    test_data = pd.read_csv('X_test.csv')
    test_data = np.array(test_data, dtype=np.float32)
    test_data = torch.tensor(test_data)
    #train_sampler = RandomSampler(test_data)
    X_test_dataloader = DataLoader(test_data, batch_size=128, num_workers=3, shuffle=False)
    preds = torch.empty((0, 6))
    for i, seq in enumerate(X_test_dataloader):
        X_seq = seq
        X_seq = X_seq.to(device)
        pred = model(X_seq)
        pred = pred.detach().cpu()
        preds = torch.cat((preds, pred))
    softmax = torch.nn.Softmax(dim=1)
    B = softmax(preds)
    C = torch.argmax(B, dim=1) + 1
    f = open('nn_full_att7.txt', mode='w')
    for i in range(len(C)):
        print(C[i].item(), file=f)
    f.close()
