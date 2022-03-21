import os 
from torchinfo import summary

def print_params_summary(model, batch_size, channels, rows, cols, verbose=1):

    dash = '='
    s = 'Pytorch Model Details'
    rep = int((100 - len(s))/2)
    
    print(dash*rep + s + dash*rep)
    
    # print(model)
    result = summary(model, input_size=(batch_size, channels, rows, cols),verbose=verbose)
    if verbose == 0:
        print(result.total_params)

def create_path(dir_path, file_name):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return os.path.join(os.path.sep, dir_path, file_name + '.pth')


def create_kernel_config():
    from torch import nn
    from model import Model
    model = Model()
    num_blocks = [1,1,1]
    out_channels = [64,128,256]
    for i in range(5,10,2):
        kernel_sizes = [i,1]
        print('Kernel:', kernel_sizes)
        model.assign_net('resnet18', num_blocks, out_channels, kernel_sizes, fourth_layer=False)
        model.prepare_data(128, 100)
        model.assign_optimizer('sgd', '0.1', lookahead=True)
        print_params_summary(model.net, 128, 3, 32, 32, verbose=0)


if __name__ == "__main__":
    create_kernel_config() 