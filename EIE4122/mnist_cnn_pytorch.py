'''
This program is based on https://github.com/deeplearningturkiye/pratik-derin-ogrenme-uygulamalari/blob/master/PyTorch/rakam_tanima_CNN_MNIST.py

'''


'''
Dataset: MNIST (http://yann.lecun.com/exdb/mnist/) 
Algorithm: Convolutional Neural Networks
'''

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

# Processing of information received from terminal command:
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--mode', type=str, default='train', 
                    help='train or test the model.')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model_path', type=str, default='models/mnist_cnn1.pth', 
                    help='The path to save your trained model.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available() # It is checked to see if there is a Cuda.

# To generate a random number:
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Importing the MNIST dataset:
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

# Creation of Convolutional Neural Networks model:
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # Input channel: 1, Output channel: 10, Filter size: 5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # Input channel: 10, Output channel: 20, Filter size: 5x5

        # We randomly drop out 50% of the neurons:
        self.conv2_drop = nn.Dropout2d() # The default dropout rate is 50%

        self.fc1 = nn.Linear(320, 50) # Number of input neurons: 320, Number of output neurons: 50
        # We have added 50 neurons in a new layer to the model.

        self.fc2 = nn.Linear(50, 10) # Number of input neurons: 50, Number of output neurons: 10
        # 10 neurons to represent our 10 classes.

    # Let's create the flowchart of the model:
    def forward(self, x):
        # Input(x) size: [1, 28, 28] x 64 (batch_size) Channel size: 1, Image size: 28x28

        # We pass the input through the "conv1" layer we defined above,
        # then we add MaxPooling layer
        # we then pass it through our ReLu activation layer:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Output size: [10, 12, 12]

        # then we add our Dropout layer that we defined above
        # then we apply MaxPooling layer,
        # finally we pass it through ReLu activation layer:
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Output size: [10, 12, 12]

        x = x.view(-1, 320) 
        # Size -1 is found by looking at the input size and other specified dimensions
        # 20x4x4 = 320:
        # Output size: [320]

        # We add 50 neurons in our "fc1" layer, which we defined above, to our model,
        # we then pass our output through our ReLu activation layer:
        x = F.relu(self.fc1(x))
        # Output size: [50]


        x = F.dropout(x, training=self.training)

        # We add 10 neurons in our "fc2" layer, which we defined above, to our model,
        x = self.fc2(x)
        # Output size: [10]
        # We got 10 outputs to represent 10 classes in our dataset.

        # Finally, we use our Softmax function to classify:
        return F.log_softmax(x)

model = Net()
if args.cuda:
    model.cuda() # Moves data to GPU.

# We create our "SGD" optimizer:
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train():
    
    tb = SummaryWriter('./log1')

    model.train() # We put our model in training mode.
    # We start the training:
    for epoch in range(1, args.epochs + 1):
        # We create our function to train the model:
        for batch_idx, (data, target) in enumerate(train_loader): # We divide the dataset into batches.
    
            if args.cuda:
                data, target = data.cuda(), target.cuda() # Moves data to GPU.
            data, target = Variable(data), Variable(target) # We are converting our data to PyTorch variables (Tensor).
            optimizer.zero_grad() 
            output = model(data) # We process the input data in our model and get our output.
            # We calculate the error by comparing the result that should be obtained with the output produced by our model:
            loss = F.nll_loss(output, target) # The negative log likelihood loss(NLLLoss)
            loss.backward() # We apply back-propagation with the error we find.
            optimizer.step() # We update our model(weights) for more optimized result.
    
            if batch_idx % args.log_interval == 0:
                
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item())) 
                    
        # add train loss to tensorboard
        tb.add_scalar("train loss", loss.item(), epoch)  
        
        # add weight histogram to tensorboard
        for name, weight in model.named_parameters():
            tb.add_histogram(name, weight, epoch)
            tb.add_histogram(f'{name}.grad',weight.grad, epoch)
            
    print('Saving CNN to %s' % args.model_path)
    torch.save(model.state_dict(), args.model_path)
    # add graph to tensorboard
    tb.add_graph(model, (data,))
        

# We create our function that will test the model:
def test():
    
    model.load_state_dict(torch.load(args.model_path))
    
    model.eval() # We put the model in test mode.
    test_loss = 0
    correct = 0
    for data, target in test_loader: # We get our test data.
        if args.cuda:
            data, target = data.cuda(), target.cuda() 
        data, target = Variable(data, volatile=True), Variable(target) # We are converting our data to PyTorch variables (Tensor).
        output = model(data) # 
        test_loss += F.nll_loss(output, target, size_average=False).item() # Calculating the batch error rate and adding it to the total error rate.
        # We calculate the error by comparing the result that should be obtained with the output produced by our model:
        pred = output.data.max(1)[1] # The result is obtained by taking the maximum probability index.
        correct += pred.eq(target.data.view_as(pred)).cpu().sum() # M

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':

    if args.mode == 'train':
        train()
        
    elif args.mode == 'test':
        test() 