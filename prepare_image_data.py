import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, USPS, FashionMNIST
from embeddings import make_embedding_resnet18
from labelling import label_transform_MNIST, label_transform_CIFAR10 , label_transform_USPS, label_transform_Fashion


ds_list = ['CIFAR10','MNIST','USPS','Fashion']

samp_size = 1000
   
for ds in ds_list:

    print(ds)
    
    if ds == 'MNIST':
        transformer = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        
        trainset = MNIST(root='./data', train=True, download=True, transform=transformer)
        testset = MNIST(root='./data', train=False, download=True, transform=transformer)         
        trainset.targets = label_transform_MNIST(trainset.targets)
        testset.targets = label_transform_MNIST(testset.targets)
        trainset.data = np.stack((trainset.data,)*3, axis=-1)
        testset.data = np.stack((testset.data,)*3, axis=-1) 
        batch_size=min(len(trainset),samp_size)
    elif ds == 'CIFAR10':
        transformer = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    
        trainset = CIFAR10(root='./data', train=True, download=True, transform=transformer)
        testset = CIFAR10(root='./data', train=False, download=True, transform=transformer)
        trainset.targets = label_transform_CIFAR10(trainset.targets)
        testset.targets = label_transform_CIFAR10(testset.targets)
        trainset.targets = torch.tensor(trainset.targets)
        testset.targets = torch.tensor(testset.targets)
        batch_size=min(len(trainset),samp_size)
    
    elif ds == 'USPS':
        transformer = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.2469,), (0.2989,))])
    
        trainset = USPS(root='./data', train=True, download=True, transform=transformer)
        testset = USPS(root='./data', train=False, download=True, transform=transformer)
        trainset.data = torch.tensor(trainset.data)
        testset.data = torch.tensor(testset.data)
        trainset.targets = torch.tensor(trainset.targets)
        testset.targets = torch.tensor(testset.targets)
        
        trainset.targets = label_transform_USPS(trainset.targets)
        testset.targets = label_transform_USPS(testset.targets)
        trainset.data = np.stack((trainset.data,)*3, axis=-1)
        testset.data = np.stack((testset.data,)*3, axis=-1) 
        batch_size=min(len(trainset),samp_size)
    elif ds == 'Fashion':
        transformer = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))])
        
        trainset = FashionMNIST(root='./data', train=True, download=True, transform=transformer)
        testset = FashionMNIST(root='./data', train=False, download=True, transform=transformer) 
        trainset.targets = label_transform_Fashion(trainset.targets)
        testset.targets = label_transform_Fashion(testset.targets)   
        trainset.data = np.stack((trainset.data,)*3, axis=-1)
        testset.data = np.stack((testset.data,)*3, axis=-1)
        batch_size=min(len(trainset),samp_size)
        
    subset1 = torch.arange(0,batch_size)
    subset2 = torch.arange(0,len(testset.data))
    
    trainset_data = trainset.data[subset1,:,:]
    testset_data = testset.data[subset2,:,:]
    y = trainset.targets[subset1].numpy()
    ytest = testset.targets[subset2].numpy()
    X = make_embedding_resnet18(trainset_data,ds=ds)
    Xtest = make_embedding_resnet18(testset_data,ds=ds)
    
    
    Xy=np.append(X,y[:,None],1)
    df=pd.DataFrame.from_records(Xy)
    
    
    file_out = "data/" + ds + "_sample" + ".csv" 
    df.to_csv(file_out)
    print(df) 

