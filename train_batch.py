from __future__ import print_function

from math import log10

import numpy
import h5py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import NetARCNN

import argparse

from datetime import datetime

logfile = "log/log_" + str(datetime.now()) + ".txt"
train_data_path = "dataset/train_data.h5"
test_data_path = "dataset/test_data.h5"
checkpoint_path = "checkpoint_ARCNN/"


parser = argparse.ArgumentParser(description="Pytorch Deblocking Example")
parser.add_argument("--batchSize", type=int, default=100, help="training batch size")
parser.add_argument("--testBatchSize", type=int, default=30, help="testing batch size")
parser.add_argument("--nEpochs", type=int, default=30, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
# Notice: it is "action" for cuda options, not type
parser.add_argument("--cuda", action="store_true", help="use cuda?") # what is "store true"?
# parser.add_argument("--threads", type=int, default=4, help="number of threads for data loader to use")
parser.add_argument("--seed",type=int, default=123, help="random seed to use.  Default = 123")
# parser.add_argument("--logfile",type=str, default=logfile, help="name of log file for training")
opt = parser.parse_args()
print(opt)



print("===> Building logfile")
output = open(logfile,'a+')
# output = open("log_"+str(datetime.now())+".txt")
# output = open("train_result.txt")
output.write("batchSize: {}\ntestBatchSize: {}\nnEpochs: {}\nlearningRate: {}\n\n".format(opt.batchSize, opt.testBatchSize, opt.nEpochs, opt.lr))
output.close()

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda") 

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

# print("===> Loading datasets")

print("===> Building model")
model = NetARCNN()
# model.load_state_dict(torch.load(checkpoint_path+".pkl"))

criterion = nn.MSELoss()

if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=opt.lr)

def train(epoch, train_data, batch_size):

    epoch_loss = 0
    iter_num = 10
    for i in range(iter_num):

        # !! iter_num is only for testing and need to be updated later

        ran = numpy.sort(numpy.random.choice(train_data['data'].shape[0], batch_size, replace=False))
        batch_data = Variable(torch.from_numpy(train_data['data'][ran,:,:,:].astype(numpy.float32)/255.0))
        batch_label = Variable(torch.from_numpy(train_data['label'][ran,:,:,:].astype(numpy.float32)/255.0))
        if cuda:
            batch_data = batch_data.cuda()
            batch_label = batch_label.cuda()

        optimizer.zero_grad()
        # Error: argument 0 is not a Variable ?
        # Maybe it is because the order of the batch_data? currently [4*32*32*1]
        # the correct order is [4*1*32*32] and I have already fixed it,
        # but the bug is still exist
        # print(batch_data)
        loss = criterion(model(batch_data), batch_label)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.6f}".format(epoch, i, iter_num, loss.data[0]))
    
    print("===> Epoch {} Complete: Avg. Loss: {:.6f}".format(epoch, epoch_loss / iter_num))
    


def test(epoch, test_data, test_batch_size):
    avg_psnr1 = 0
    avg_psnr2 = 0
    iter_num = 5
    for i in range(iter_num):

        # !! iter_num is only for testing and need to be updated later

        ran2 = numpy.sort(numpy.random.choice(test_data['data'].shape[0],test_batch_size, replace=False))
        batch_data = Variable(torch.from_numpy(test_data['data'][ran2,:,:,:].astype(numpy.float32)/255.0))
        batch_label = Variable(torch.from_numpy(test_data['label'][ran2,:,:,:].astype(numpy.float32)/255.0))
        if cuda:
            batch_data = batch_data.cuda()
            batch_label = batch_label.cuda()

        prediction = model(batch_data)
        
        mse1 = criterion(batch_data, batch_label)
        mse2 = criterion(prediction, batch_label)
        psnr1 = 10 * log10(1 / mse1.data[0])
        psnr2 = 10 * log10(1 / mse2.data[0])

        avg_psnr1 += psnr1
        avg_psnr2 += psnr2
    print("===> Before: Avg. PSNR: {:.5f} dB; after: Avg. PSNR: {: .5f} dB".format((avg_psnr1 / iter_num), (avg_psnr2 / iter_num)))
    # print the result to a file so as to track the tendency
    # output = open('train_result.txt','a+')
    output = open(logfile, 'a+')
    output.write("{} {: .5f} {: .5f}\n".format(epoch, (avg_psnr1/iter_num), (avg_psnr2/iter_num)))
    output.close()
def checkpoint(epoch):
    # model_out_path = "model_epoch_{}.pth".format(epoch) # is that all right? 
    # torch.save(model, checkpoint_path+model_out_path)
    torch.save(model.state_dict(), checkpoint_path+("checkpoint_epoch_{}.pkl".format(epoch))) # use this line to save parameters only 
    print("Checkpoint saved to {}".format(checkpoint_path))

# can I open 2 files at one time? 
# A: Sure

train_data = h5py.File(train_data_path, "r")
test_data = h5py.File(test_data_path, "r")


for epoch in range(1, opt.nEpochs + 1):
    train(epoch, train_data, opt.batchSize)
    test(epoch, test_data, opt.testBatchSize)
    checkpoint(epoch)

