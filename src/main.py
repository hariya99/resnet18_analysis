import argparse
from torch import nn
from model import Model
from utils import * 

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
    parser.add_argument('-pf', metavar="Param File", default="params", type=str, 
                        help='Params file name to store params')
    parser.add_argument('-ini', metavar="Weight Init", default="he", type=str, 
                        help='Weight Initializer')
    args = parser.parse_args()
    print("Current Args: ", args)
    return args
    # parsing ends 

def main():
    args = parse_args()
    path = create_path("saved_params", args.pf)
    model = Model()
    
    # config-1
    # num_blocks = [2,1,1,1]
    # conv_channels = 64
    # kernel_sizes = [3,1]

    # Config-2
    # num_blocks = [1,1,1]
    # conv_channels = 128
    # kernel_sizes = [3,1]

    # Config-3 - Best so far
    num_blocks = [4,3,3]
    conv_channels = 64
    kernel_sizes = [3,1]

    model.assign_net(args.m, num_blocks, conv_channels, kernel_sizes)
    model.prepare_data(128, 128, args.dl)
    model.assign_optimizer(args.o, args.lr, lookahead=True)

    # initialize weights of linear layer 
    model.init_weights(init_type=args.ini)

    for epoch in range(args.e): 
        model.train()
        model.test()
        model.scheduler.step()
        if model.best_accuracy < model.test_accuracy_list[-1]:
            # save parameters
            model.best_accuracy = model.test_accuracy_list[-1] 
            model.save_params(epoch,path)
    
    # print statistics 
    model.print_stats()
    
    # plot the statistics 
    # model.plot_stats()

    # print summary of the model and its parameters
    # print_params_summary(model.net, 128, 3, 32, 32)

if __name__ == "__main__":
    main()