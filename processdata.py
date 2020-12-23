import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import torch


def read_input_fc(ifilename = "traindata.idx3"): #reads img file and label files and returns arrays
    images = idx2numpy.convert_from_file(ifilename)  # variable will store images in 3-D array
    imgdata = np.reshape(images, newshape=[images.shape[0], -1])/255
    imgdata = torch.from_numpy(imgdata)
    return imgdata

def read_input_cnn(ifilename = "traindata.idx3"):
    images = idx2numpy.convert_from_file(ifilename)
    images = images/255
    images = torch.from_numpy(images)
    return images

def plot_graph(loss_epoch, loss_batch):
    plt.plot(loss_epoch)
    plt.ylabel("Loss")
    plt.xlabel("No of EPOCHS")
    plt.show()
    plt.clf()
    plt.plot(loss_batch)
    plt.ylabel("Loss")
    plt.xlabel("No of Batches")
    plt.show()
    plt.clf()
    return
