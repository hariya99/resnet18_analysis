import argparse
from torchinfo import summary
from torch import nn
from model import Model

def parse_args():
    
    # parsing start
    parser = argparse.ArgumentParser(description="CIFAR10")
    parser.add_argument("-dl", metavar="DataLoadWorker", default=2, type=int, 
                        help="# of Data Loader Workers")
    parser.add_argument("-o",  metavar="Optimizer", default="sgd", type=str, help="Optimizer")
    parser.add_argument('-lr', metavar="LearningRate", default=0.1, type=float, help='learning rate')
    parser.add_argument('-e', metavar="Epoch", default=5, type=int, help='# of epoch')
    parser.add_argument('-m', metavar="Model", default="resnet18", type=str, 
                        help='Model you want to use e.g. resnet18')

    args = parser.parse_args()
    print("Current Args: ", args)
    return args
    # parsing ends 
    
def print_params_summary(model, batch_size, channels, rows, cols):

    dash = '='
    s = 'Pytorch Model Details'
    rep = int((100 - len(s))/2)
    
    print(dash*rep + s + dash*rep)
    print(model)
    summary(model, input_size=(batch_size, channels, rows, cols))

def init_weights(model, init_type="normal"):
    '''
        Initialize parameters of linear layer
    '''
    if init_type == "normal":
        nn.init.normal_(model.net.linear.weight, mean=0, std=0.01)
    elif init_type == "xavier":
        nn.init.xavier_uniform_(model.net.linear.weight)
    else:
        return

def main():
    args = parse_args()

    model = Model()
    num_blocks = [2,1,1,1]
    out_channels = [64,128,256,512]

    model.assign_net(args.m, num_blocks, out_channels)
    model.prepare_data(128, 100, 2)
    model.assign_optimizer(args.o, args.lr)

    # initialize weights of linear layer 
    init_weights(model)
    
    for epoch in range(args.e): 
        model.train()
    
    model.test()
    
    # print statistics 
    model.print_stats()
    
    # plot the statistics 
    # model.plot_stats()

    # print summary of the model and its parameters
    # print_params_summary(model.net, 128, 3, 32, 32)

if __name__ == "__main__":
    main()