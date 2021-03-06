# torch imports 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

# local imports
from cnns import ResNet18
from lookahead import Lookahead

class Model:
    '''
        Model class has all the code needed to invoke a model
        Usage: 
        model = Model()
        model.assign_net(net) e.g resnet18
        model.prepare_data(train_batch, test_batch, workers)
        model.assign_optimizer(optimizer, lr) e.g. sgd
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
        self.train_loss_list = []
        self.train_accuracy_list = []
        self.test_loss_list = []
        self.test_accuracy_list = []
        self.best_accuracy = 0.0
        self.best_epoch = 0
        
    def assign_net(self, net='resnet18', 
                    blocks_list=[2,2,2,2], 
                    conv_channels=64,
                    kernel_sizes=[3,1]):
        ''' 
            net: provide the type of neural net you want to run. 
            blocks_list: list of number of blocks in each layer. 
            conv_channels: first layer channels.
            pool_kernel_sz: average pool kernel size.
        '''
        if(net.lower() == 'resnet18'):
            self.net = ResNet18(blocks_list, conv_channels, kernel_sizes)
            self.net.to(self.device)

            # parallelize the network training 
            if self.device == 'cuda':
                self.net = torch.nn.DataParallel(self.net)
                cudnn.benchmark = True

    def init_weights(self, init_type="normal"):
        '''
            Initialize parameters of linear layer
        '''
        def _apply_init(m):
            if (isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d)) and hasattr(m, 'weight'):
                if init_type == "normal":
                    nn.init.normal_(m.weight.data, mean=0, std=0.5)
                elif init_type == "xavier":
                    nn.init.normal_(m.weight.data)
                elif init_type == 'he':
                    nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
                else:
                    pass
        
        self.net.apply(_apply_init)
                

    
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

    def assign_optimizer(self, optimizer, lr, lookahead=True):
        # choose optimizer
        if (optimizer.lower() == "sgdn"):
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4, nesterov=True)
        elif (optimizer.lower() == "adagrad"):
            self.optimizer = optim.Adagrad(self.net.parameters(), lr=lr, weight_decay=5e-4)
        elif (optimizer.lower() == "adadelta"):
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr, weight_decay=5e-4)
        elif (optimizer.lower() == "adam"):
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=5e-4)
        else:
            # default
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
        
        # lookahead 
        if lookahead:
            self.optimizer = Lookahead(self.optimizer, k=5, alpha=0.5) # Initialize Lookahead
        self.scheduler = self._set_scheduler()


    def _set_device(self):
        return 'cuda' if (torch.cuda.is_available()) else 'cpu'

    def _set_criterion(self):
        return nn.CrossEntropyLoss()

    def _set_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)

    def preprocess(self, images, targets):
        batch_size = images.shape[0]
        images_90 = torch.flip(images.transpose(2, 3), (2,))
        images_180 = torch.flip(images, (2, 3))
        images_270 = torch.flip(images, (2,)).transpose(2, 3)

        images_batch = torch.cat((images, images_90, images_180, images_270), dim=0)
        # targets = torch.arange(4).long().repeat(batch_size)
        # targets = targets.view(batch_size, 4).transpose(0, 1)
        # targets = targets.contiguous().view(-1)
        targets = targets.repeat(4)
        return images_batch, targets

    def pre_train(self):

        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
       
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            
            pre_inputs, pre_targets = self.preprocess(inputs, targets)
            pre_inputs, pre_targets = pre_inputs.to(self.device), pre_targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(pre_inputs)
            loss = self.criterion(outputs, pre_targets)
            loss.backward()
            self.optimizer.step()
            # train_loss += loss.item()
            # _, predicted = outputs.max(1)
            # total += pre_targets.size(0)
            # correct += predicted.eq(pre_targets).sum().item()

        
        # self.train_loss_list.append(train_loss/(batch_idx + 1))
        # self.train_accuracy_list.append((correct/total) * 100)

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
        
        self.train_loss_list.append(train_loss/(batch_idx + 1))
        self.train_accuracy_list.append((correct/total) * 100)


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
        # self._print_stats(f'Correct|Total : {correct}|{total}', 'test',
        #             test_loss/len(self.test_loader), correct/total)
        
        self.test_loss_list.append(test_loss/(batch_idx + 1))
        self.test_accuracy_list.append((correct/total) * 100)

    def save_params(self, epoch, path):
        # Save checkpoint.
        # print('Saving Params..')
        state = {
            'net': self.net.state_dict(),
            'best_acc' : self.best_accuracy,
            'train_acc' : self.train_accuracy_list,
            'test_acc' : self.test_accuracy_list,
            'train_loss' : self.train_loss_list,
            'test_loss' : self.test_loss_list,
            'epochs' : epoch 
        }
        torch.save(state, path)

    def load_params(self, path):
        # Load params
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net'])
        self.best_accuracy = checkpoint['best_acc']
        self.best_epoch = checkpoint['epochs']
        self.train_accuracy_list = checkpoint['train_acc']
        self.test_accuracy_list = checkpoint['test_acc']
        self.train_loss_list = checkpoint['train_loss']
        self.test_loss_list = checkpoint['test_loss']

    def print_stats(self):
        dash = '*'
        s = 'Model Statistics'
        rep = int((100 - len(s))/2)
        print(dash*rep + s + dash*rep)
        print(f"Train Loss     : {min(self.train_loss_list)}")
        print(f"Train Accuracy : {max(self.train_accuracy_list)}")
        print(f"Test Loss      : {min(self.test_loss_list)}")
        print(f"Test Accuracy  : {max(self.test_accuracy_list)}")


    def plot_stats(self, plot_name):
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,7))
        fig.suptitle('Error, Accuracy Plots')
        axs[0, 0].plot(range(len(self.train_loss_list)), self.train_loss_list)
        axs[0, 0].set_title('Train Loss')
        axs[0, 0].text(0.9,0.6,f'Min Loss {min(self.train_loss_list):.4f}', 
                       horizontalalignment='right', verticalalignment='top', transform=axs[0, 0].transAxes)
        
        axs[0, 1].plot(range(len(self.train_accuracy_list)), self.train_accuracy_list, 'tab:orange')
        axs[0, 1].set_title('Train Accuracy')
        axs[0, 1].text(0.9,0.6,f'Max Accuracy {max(self.train_accuracy_list):.2f}', 
                       horizontalalignment='right', verticalalignment='top', transform=axs[0, 1].transAxes)
            
        axs[1, 0].plot(range(len(self.test_loss_list)), self.test_loss_list, 'tab:green')
        axs[1, 0].set_title('Test Loss')
        axs[1, 0].text(0.9,0.6,f'Min Loss {min(self.test_loss_list):.4f}', 
                       horizontalalignment='right', verticalalignment='top', transform=axs[1, 0].transAxes)
            
        axs[1, 1].plot(range(len(self.test_accuracy_list)), self.test_accuracy_list, 'tab:red')
        axs[1, 1].set_title('Test Accuracy')
        axs[1, 1].text(0.9,0.6,f'Max Accuracy {max(self.test_accuracy_list):.2f}', 
                       horizontalalignment='right', verticalalignment='top', transform=axs[1, 1].transAxes)
        plt.tight_layout()
        plt.savefig(f"{plot_name}.pdf")