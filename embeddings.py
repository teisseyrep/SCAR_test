import numpy as np
import torch
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable


def make_embedding_resnet18(trainset_data,ds='CIFAR10'):
   
    # Load the pretrained model
    model = models.resnet18(pretrained=True)
    # Use the model object to select the desired layer
    layer = model._modules.get('avgpool')
    model.eval()
    if ds=='CIFAR10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.247, 0.243, 0.261])
    elif ds=='MNIST':
        normalize =transforms.Normalize((0.1307,), (0.3081,))
    elif ds=='USPS':
        normalize = transforms.Normalize((0.2469,), (0.2989,))
    elif ds=='Fashion':
        normalize = transforms.Normalize((0.2860,), (0.3530,))
    else:
        raise Exception("ds not found!")
        
    to_tensor = transforms.ToTensor()
   
    n = len(trainset_data)
    X = np.zeros((n,512))
    for i in np.arange(0,n):
        if i % 1000==0 and i>0:
            print(i)
        img = trainset_data[i]   
        # 1. Create a PyTorch Variable with the transformed image
        t_img = Variable(normalize(to_tensor(img)).unsqueeze(0))
        # 3. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        my_embedding = torch.zeros(512)
        # 4. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            my_embedding.copy_(o.data.reshape(o.data.size(1)))
        # 5. Attach that function to our selected layer
        h = layer.register_forward_hook(copy_data)
        # 6. Run the model on our transformed image
        model(t_img)
        # 7. Detach our copy function from the layer
        h.remove()
        # 8. Return the feature vector 
        X[i,:]=my_embedding.numpy()
    
    return(X)
