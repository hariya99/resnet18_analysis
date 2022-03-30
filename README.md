# ResNet18_analysis

Analyze Resnet-18 for all the possible configurations like depth, width, kernel sizes etc. to bring down the number of learnable parameters from ~11M to under 5M but also to explore various optimization techniques so that it performs at the same or better level of accuracy on CIFAR-10 dataset.

# Best Configuration Identified

| Configuration                 | Optimal Model             | ResNet-18                |
| :---                          |    :----:                 |          ---:             |
|Number of residual layers      | 3                         | 4|
|Number of residual blocks      | [4, 4, 3]                 | [2, 2, 2, 2] |
|Convolutional kernel sizes     | [3, 3, 3]                 | [3, 3, 3, 3]|
|Shortcut kernel sizes          | [1, 1, 1]                 | [1, 1, 1, 1] |
|Number of channels             | [64, 128, 256]            | [64, 128, 256, 512]   |
|Average pool kernel size       | 8                         | 4 |
|Batch normalization            | True                      | True |
|Dropout                        | 0                         | 0|
|Squeeze and excitation         | True                      | False|
|Gradient clip                  | 0.1                       | None|
|Data augmentation              | True                      | False  |
|Data normalization             |   True                    | False|
|Lookahead                      | True                      | False|
|Optimizer                      | SGD                       | SGD|
|Learning rate (lr)             | 0.1                       | 0.1|
|Lr scheduler                   | CosineAnnealingLR         | CosineAnnealingLR|
|Weight decay                   | 0.0005                    | 0.0005 |
|Batch size                     | 128                       | 128|
|Number of workers              | 2                         | 2   |
|Total number of Parameters     | 4,697,742                 | 11,173,962 |