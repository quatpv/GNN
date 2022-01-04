import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from models import Model
from torch.utils.data import DataLoader
from dataset.loader import get_training_set, get_validation_set, get_test_set
# from torch_geometric.loader import DataLoader
# from torch_geometric.datasets import TUDataset
from pprint import pprint

parser = argparse.ArgumentParser()

# For GNN
parser.add_argument('--seed', type=int, default=7497, help='random seed')
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
parser.add_argument('--pooling_ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='ucf101', help='dataset name')
parser.add_argument('--device', type=str, default='cpu', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=2, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')

parser.add_argument('--num_classes', type=int, default=101, help='number of classes')
parser.add_argument('--num_features', type=int, default=1024, help='number of features')

# For video streaming
parser.add_argument('--video_path', type=str, default='dataraw/UCF-101/video', help='path to video file')
parser.add_argument('--dataset_file', type=str, default='dataraw/UCF-101/ucf101_01.json', help='train/val split')
parser.add_argument('--sample_duration', type=int, default=64, help='temporal duration of inputs')
parser.add_argument('--stride_size', type=int, default=4, help='temporal stride of inputs')
parser.add_argument('--sample_size', type=int, default=224, help='height and width of inputs')
parser.add_argument('--n_threads', type=int, default=4, help='Number of threads for multi-thread loading')


args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


training_data = get_training_set(args)
train_loader = DataLoader(training_data,
                          batch_size=args.batch_size,
                          shuffle=True,
                          pin_memory=True,
                          drop_last=True, 
                          sampler=None)

validation_data = get_validation_set(args)
val_loader = DataLoader(validation_data,
                        batch_size=args.batch_size,
                        shuffle=False,
                        pin_memory=True)

model = Model(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train():
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()
    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        for i, (inputs, targets, index) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            out = model(inputs)
            loss = F.nll_loss(out, targets)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            pred = out.max(dim=1)[1]
            print("pred: ", pred)
            print("labels: ", targets)
            correct += pred.eq(targets).sum().item()
            print("correct: ", correct)
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = compute_test(val_loader)
        print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
              'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
              'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t))

        val_loss_values.append(loss_val)
        torch.save(model.state_dict(), 'weights/{}.pth'.format(epoch))
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('weights/*.pth')
        for f in files:
            epoch_nb = int(f[8:].split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('weights/*.pth')
    for f in files:
        epoch_nb = int(f[8:].split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time.time() - t))

    return best_epoch


def compute_test(loader):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()
    return correct / len(loader.dataset), loss_test


if __name__ == '__main__':
    # Model training
    best_model = train()

    # Restore best model for test set
    # model.load_state_dict(torch.load('weights/{}.pth'.format(best_model)))
    # test_acc, test_loss = compute_test(test_loader)
    # print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))

