import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import processdata
import matplotlib.pyplot as plt
import numpy
import cv2

# constants
lr = 0.0008
epoch = 8
batchsize = 64
imgcheck = 1000  # Checks for that image's index in the test set(0-9999)
# use imgcheck b/w 0-9990 because it shows 10 images starting from index of imgcheck
modelname = "cnn2.pt"


def read_data():
    traindata = processdata.read_input_cnn()
    print("Train data size is: ", traindata.shape[0])  # shows number of images in train set
    return traindata


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Conv2d(1, 8, 6, 2)
        self.l2 = nn.Conv2d(8, 32, 4, 4, 2)

        self.l3 = nn.Flatten()  # Flattened into a 256-D vector
        self.bottle = nn.Linear(512, 200)
        self.fc2 = nn.Linear(200, 512)

        self.rl1 = nn.ConvTranspose2d(32, 8, 4, 4, 2)
        self.rl2 = nn.ConvTranspose2d(8, 1, 6, 2)
        return

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        x = self.l3(x)
        x = F.relu(self.bottle(x))
        x = F.relu(self.fc2(x))
        x = x.reshape((-1, 32, 4, 4))

        x = F.relu(self.rl1(x))
        x = torch.sigmoid(self.rl2(x))
        return x


def dec_network(traindata):  # declare network
    net = Network()
    net.double()  # prevents an error
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    pred = net(traindata)
    loss = loss_func(pred.squeeze(), traindata)
    print("Initial Loss is " + str(loss.item()))
    return net, optimizer, loss_func, loss.item()


def fit(net, traindata, optimizer, loss_func, loss_init):
    loss_batch = []
    loss_epoch = [loss_init]
    for i in range(epoch):
        for j in range(int(traindata.shape[0]/batchsize)):
            x_batch = traindata[j*batchsize:(j+1)*batchsize]
            optimizer.zero_grad()
            pred = net(x_batch)
            loss = loss_func(pred.squeeze(), x_batch)
            loss_batch.append(loss.item())
            loss.backward()  # model learns by backpropagation
            optimizer.step()  # model updates its parameters
            if (j+1) % 100 == 0:
                print("EPOCH No: ", i+1, " ", (j+1), " Batches done")
        pred = net(traindata)
        loss = loss_func(pred.squeeze(), traindata)
        loss_epoch.append(loss.item())
        print("Loss after EPOCH No " + str(i+1) + ": " + str(loss.item()))  # prints loss
        del loss
    return loss_epoch, loss_batch




def main():
    traindata = read_data()
    net, optimizer, loss_func, loss_init = dec_network(traindata)
    need_train = input("Enter 'm' to train model, anything else to load old model: ")
    if need_train == 'm' or need_train == 'M':
        loss_epoch, loss_batch = fit(net, traindata, optimizer, loss_func, loss_init)
        processdata.plot_graph(loss_epoch, loss_batch)
        need_save = input("Enter 's' to save model, anything else to not save: ")
        if need_save == 's' or need_save == 'S':
            print("Saving Model...")
            torch.save(net.state_dict(), modelname)
    else:
        net.load_state_dict(torch.load(modelname))
    testdata = processdata.read_input_cnn('testdata.idx3')
    print("Original images are: ")
    img = numpy.asarray(testdata[imgcheck].squeeze())
    for i in range(1, 10):
        pic = numpy.asarray(testdata[imgcheck + i].squeeze())
        img = cv2.hconcat([img, pic])
    plt.axis('off')
    plt.imshow(img, cmap='Greys_r')
    plt.show()
    pred = net(testdata)
    loss = loss_func(pred.squeeze(), testdata)
    print("Final Loss on Test set is: " + str(loss.item()))
    print("Regenerated images are: ")
    img = pred[imgcheck].squeeze().detach().numpy()
    for i in range(1, 10):
        pic = pred[imgcheck+i].squeeze().detach().numpy()
        img = cv2.hconcat([img, pic])
    img = img.squeeze()
    plt.axis('off')
    plt.imshow(img, cmap='Greys_r')
    plt.show()
    return

