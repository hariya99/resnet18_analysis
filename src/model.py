# torch imports 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

# local imports
from cnns import ResNet18

class Model:
    '''
        Model class has all the code needed to invoke a model
        Usage: 
        model = Model()
        model.assign_net(net) e.g resnet18
        model.prepare_data(train_batch, test_batch, workers)
        model.assign_optimizer(optimizer) e.g. sgd
        loop for epochs 
            model.train()
        model.test()


    '''
    def __init__(self):
        self.train_loader = None
        self.test_loader = None
        self.optimizer= None
        self.scheduler = None 
        self.net = None
        self.device = self._set_device()
        self.criterion = self._set_criterion()
        # self.batch_size = batch_size
        # self.workers = workers
        
    def assign_net(self, net='resnet18'):
        if(net.lower() == 'resnet18'):
            self.net = ResNet18()

    def prepare_data(self, train_batch=128, test_batch=100, workers=2):
        '''
            Supply train, test batch sizes and number of data workers
        '''
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True, transform=transform_train)

        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch, shuffle=True, num_workers=workers)

        testset = torchvision.datasets.CIFAR10(
            root='../data', train=False, download=True, transform=transform_test)

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=test_batch, shuffle=False, num_workers=workers)

    def assign_optimizer(self, optim):
        # choose optimizer
        if (optim.lower() == "sgdn"):
            self.optimizer = optim.SGD(net.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)
        elif (optim.lower() == "adagrad"):
            self.optimizer = optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=5e-4)
        elif (optim.lower() == "adadelta"):
            self.optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=5e-4)
        elif (optim.lower() == "adam"):
            self.optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
        else:
            # default
            self.optimizer = optim.SGD(net.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
        self.scheduler = self._set_scheduler()

    def _set_device(self):
        return 'cuda' if (torch.cuda.is_available()) else 'cpu'

    def _set_criterion(self):
        return nn.CrossEntropyLoss()

    def _set_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
# Training
def train(self):

    self.net.train()
    train_loss = 0
    correct = 0
    total = 0

    
    for batch_idx, (inputs, targets) in enumerate(self.train_loader):
        
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    # print statistics 
    self._print_stats(f'Correct|Total : {correct}|{total}', 
                train_loss/len(self.train_loader), correct/total)


def test(self):

    self.net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(self.test_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # print statistics 
    self._print_stats(f'Correct|Total : {correct}|{total}', 
                train_loss/len(self.test_loader), correct/total)

def _print_stats(msg, train_loss, train_acc):
    print("*****Epoch Statistics*****")
    print(msg)
    print("Epoch training loss: ", train_loss)
    print("Epoch training accuracy: ", train_acc)