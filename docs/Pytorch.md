## Pytorch
1. In pytorch for loading a custom dataset one must inherit the __Dataset__ class. From __torch.utils.data import Dataset__ .
2. After inheriting the class two methods must be implemented for the custom dataset class to work.
3. Transformation operations, While loading the dataset we can apply transforms operations on it which is simply a list of functions that are applied on the dataset. Like pytorch accetps the inputs as Tensor objects only so we can convert x_train into Tensors so __transforms.ToTensor, transforms.Normalize([list of mean], [list of stddev])__ and much more.
4. \__getitem__(self, idx) method which must return the input and the label. And the other method implemented must be \__len__().
5. After inhereting the dataset class , it's simply not enough so we must use __DataSetLoader__ class which acts as a wrapper around our dataset and provides useful functionalities like generating dataset in a fixed batch size, shuffling the data.
6. another useful feature is the torchvision module which is specifically made for image related operations. __torchvision.datasets.ImageFolder(root, transform=None)__, x = datasets.ImageFolder(root), then __x.imgs__ is a tuple of the absolute image path and its class so we can return the various images after maybe shuffling the dataset.
7. Given an image path, to read the whole image we can use the function __img = cv2.imread(path)__ and __cv2.imshow(img)__ to display the image.

## General pytorch training Path

### Create the network
we can create the network by inhereting the __torch.nn.Module__  class

class Net(nn.Module):
  def \__init__(self, parameters=None):
    self.parameters = parameters # any extra params
    self.cnn1 = nn.Sequential(
      nn.Conv2d(inp_channels, out_channels,kernel, stride),
      .....
    )
    self.linear = nn.Linear(in_feat, out_feat)
  def forward(self, inp):
    #this method must be implemented
    import torch.nn.functinal as F
    F.relu()
    F.max_pool2d()
    etc
    and remember to flatten the inputs before passing it on to the linear layer

### Create the Custom Dataset class
As discussed above a Custom dataset class can be created if needed or we can use torchvision's inbuilt modules like ImageFolderDataset with the data set in a fixed layout

### Create Custom loss functions
After reading a bit , I understood that CustomLoss functions are nothing but simply inherited networks from nn.Module . Here while implementing forward method we pass the target and label as input as compared to inp and the \__call()__ method implements self.forward(*args, \**kwargs) like in the net class
**ALWAYS IT IS TO BE KEPT IN MIND THAT pytorch accepts Tensors as input for inputs and labels both and the input is always in the form of (batch_size, channels, height, width)**

### Defining the optimizers and criterion
After the custom loss function is implemented , we need to define the optimizers and the loss criterion.
optimizers are used for kind of defining the rules to update the paratmeters of the network.
Examples of various optimizers are SGD, Adam, etc. if we dont use optimizers then we will need to update the parameters of the network ourselves. after calling **loss.backward()** method.
**optimizer = torch.optim.SGD(net.parameters(), lr=.001)**
Here net.parameters() are the parameters of the neural network we defined i.e. the parameters on which the gradients will be computed while calling the backwards method.
we can simply call **optimizer.step()** to update the weights

### Training Loop

for epoch:
for i, (imgs, labels) in enumerate(trainloader):

calulate the output
**target = net(imgs)**
calulate the loss
**loss = criterion(target, labels)**
zero the grad from prev loop
**optimizer.zero_grad()**
calulate the grads
**loss.backward()**
update the weights
**optimizer.step()**
one can also do additional things like calculating the loss at every epoch
**running_loss = loss.item() * imgs.size(0)**

**loss_tot = running_loss / len(trainloader.dataset)**

### Testing loop

to do ...