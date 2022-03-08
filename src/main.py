import argparse
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

def main():
    args = parse_args()

    model = Model()
    model.assign_net(args.m)
    model.prepare_data(128, 100, 2)
    model.assign_optimizer(args.o, args.lr)

    for epoch in range(args.e): 
        model.train()
    
    model.test()

if __name__ == "__main__":
    main()