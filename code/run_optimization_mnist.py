import sys

if len(sys.argv) != 5:
    raise ValueError()

params = sys.argv[1:]
GPU_IDX = params[0]
N = int(params[1])
num_try = int(params[2])
n_epochs = int(params[3])

import os 
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_IDX



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

print(GPU_IDX, torch.cuda.is_available())

import numpy as np
import tqdm

from scipy.misc import imresize

from torchvision import transforms

torch.set_default_tensor_type('torch.FloatTensor')

from mnist_data import load_dataset


def onehot_labels(a):
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b


class myDataSet(Dataset):
    def __init__(self, objects, labels, transform=None):
        assert len(objects) == len(labels)
        self.X = objects
        self.y = labels
        self.len = len(objects)
        self.transform = transform
    
    def __getitem__(self, idx):
        sample = self.X[idx], self.y[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.len


class Downsample(object):
    def __init__(self, p_down):
        self.p_down = p_down
    
    def __call__(self, sample):
        image, label = sample
        restored_image = image.reshape(28,28)
        image = imresize(restored_image, self.p_down, mode='F').ravel()
        return image, label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample

        return torch.from_numpy(image).double(), torch.from_numpy(label).double()


def get_predictions_on_dataset(model, dataset, batch_size=None, compute_accuracy=False, num_output=None):
    def _target_predictions(output, num_output):
        if num_output is None:
            return output
        else:
            return output[num_output]
        
    if batch_size is None:
        batch_size = len(dataset)
    data_loader = DataLoader(dataset, 
                             batch_size=batch_size,
                             shuffle=False, 
                             num_workers=2)
    raw_predictions = np.zeros((len(dataset), dataset[0][1].shape[0]))
    all_predictions = np.zeros(len(dataset))
    all_labels = np.zeros_like(all_predictions)
    
#     print(batch_size)
    
    with torch.no_grad():
        cur_start = 0
        for batch, labels in data_loader:
            _raw_prediction = _target_predictions(model(batch.cuda()), num_output)
            raw_predictions[cur_start:cur_start+batch_size] = np.array(_raw_prediction)
#             print(np.array(_raw_prediction).shape)
            predictions = np.array(_raw_prediction)
#             print(predictions.shape)
            predictions = np.argmax(predictions, axis=1)
            labels = np.argmax(np.array(labels), axis=1)
            all_predictions[cur_start:cur_start+batch_size] = predictions
            all_labels[cur_start:cur_start+batch_size] = labels
            cur_start += batch_size
    if compute_accuracy:
        val_acc = np.mean(all_labels==all_predictions, dtype='float')
        return raw_predictions, all_labels, val_acc
    else:
        return raw_predictions, all_labels


def cross_entropy(_input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        _input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        _input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(_input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(-1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(_input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(_input), dim=1))


class TwoOutputsNN(torch.nn.Module):
    def __init__(self, d, m, q):
        super(TwoOutputsNN, self).__init__()
        self._lin1 = torch.nn.Linear(d, m)
        self._act1 = torch.nn.ReLU()
        self._lin2 = torch.nn.Linear(m, m)
        self._act2 = torch.nn.ReLU()
        self._lin3 = torch.nn.Linear(m, q)
        self._out_softmax = torch.nn.Softmax(-1)
        
        self._queue = [
            self._lin1,
            self._act1,
            self._lin2,
            self._act2,
            self._lin3
        ]
        
    def forward(self, x):
        result = x
        for layer in self._queue:
            result = layer(result)
        
        out1 = self._out_softmax(result)
        out2 = result
        return out1, out2
    
def get_teacher(d, m, q):
    model = TwoOutputsNN(d, m, q)
    opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    
    loss = cross_entropy
    
    return (model, opt, loss)


def Dist(d, m, q, L):
    def _hard_loss(_input, target, L):
        return (1.-L)*cross_entropy(_input, target)

    def _soft_loss(_input, target, L):
        return L*cross_entropy(_input, target)

    
    model = torch.nn.Sequential()
    model.add_module('d1', torch.nn.Linear(d, m))
    model.add_module('a1', torch.nn.ReLU())
    model.add_module('d2', torch.nn.Linear(m, m))
    model.add_module('a2', torch.nn.ReLU())
    model.add_module('d3', torch.nn.Linear(m, q))
    model.add_module('a3', torch.nn.Softmax(-1))
    
    hard_loss = _hard_loss
    soft_loss = _soft_loss

    
#     opt = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    
    
    return (model, opt, hard_loss, soft_loss)


def _get_iterator_wrapper(iterable, _tqdm=False, leave=False, desc=''):
    if _tqdm:
        return tqdm.tqdm(iterable, leave=leave, desc=desc)
    else:
        return iterable


def main():
    saved_filename = 'saved_objects_mnist_{}.pcl'.format(N)
    from_save = torch.load(saved_filename)

    batch_size = 50
    m = 20
    q = 10
    d = from_save['transformed_data_train'][0][0].shape[0]

    model_predictions = from_save['model_predictions']
    logfile_name = 'torch_version_logs/mnist_{}/new_log_mnist_{}.txt'.format(N, num_try)
    iofile = open(logfile_name, 'w')

    for T in _get_iterator_wrapper([1,2,5,10,20,50], _tqdm=False, desc='T_loop'):
        print('{}, {} started with T = {}'.format(N, num_try, T))

        for L in _get_iterator_wrapper([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], _tqdm=False, leave=False, desc='L_loop'):                
            labels_soften = F.softmax(model_predictions/T, -1)
            student, student_opt, hard_loss, soft_loss = Dist(d, m, q, L)
            student.double()
            student.cuda()

            _t = _get_iterator_wrapper(np.arange(n_epochs))
            for epoch in _t:
                cur_start = 0
                loss_hist = []
                for batch, label in _get_iterator_wrapper(from_save['transformed_dataloader_train']):
            #         print(cur_start, batch_size, labels_soften[cur_start:cur_start+batch_size].shape)
                    batch = batch.cuda()
                    label = label.cuda()
                    # Step 1. Remember that PyTorch accumulates gradients.
                    # We need to clear them out before each instance
                    student.zero_grad()

                    predictions = student(batch)
                    hard_loss = cross_entropy(predictions, label)
                    soft_loss = cross_entropy(predictions, labels_soften[cur_start:cur_start+batch_size].cuda())

                    total_loss = (1.-L)*hard_loss + L*soft_loss
                    total_loss.backward()
            #                 soft_loss.backward()
    #                 loss_hist.append(np.mean(np.array(total_loss.detach())))


                    student_opt.step()
                    cur_start += batch_size

    #             if epoch % 25 == 0:
    #                 val_acc = get_predictions_on_dataset(student, from_save['transformed_data_val'], compute_accuracy=True)[-1]
    #                 print(val_acc)
    #             _t.set_postfix(val_acc=val_acc, mean_loss = np.mean(loss_hist[-50:]))

            acc_student = get_predictions_on_dataset(student, from_save['transformed_data_test'], compute_accuracy=True)[-1]
            iofile.write(str([N, T, L, acc_student])+'\n')

    iofile.close()

    return 'Process with N = {}, num_try = {}, n_epochs = {} finished. Log file {}'.format(N, num_try, n_epochs, logfile_name)

if __name__ == '__main__':
    res = main()
    print(res)
