from torchinfo import summary

def print_params_summary(model, batch_size, channels, rows, cols):

    dash = '='
    s = 'Pytorch Model Details'
    rep = int((100 - len(s))/2)
    
    print(dash*rep + s + dash*rep)
    print(model)
    summary(model, input_size=(batch_size, channels, rows, cols))