import torch


def find_zero_fitler(param):
    # param = param.copy()
    param = param.view(param.shape[0], -1)
    zero_num = 0
    for i in range(param.shape[0]):
        if torch.sum(torch.abs(param[i, :])) == 0:
            zero_num = zero_num + 1
    return zero_num


def analyse_network(model):
    for name, param in model.state_dict().items():
        zero_num = find_zero_fitler(param)
        if zero_num != 0:
            if name[-4:] != 'bias' and name[-4:] != 'mean' and name[-3:] != 'var':
                print("name: " + name + " zero: " + str(zero_num))
            # print("name: " + name + " zero: " + str(zero_num))
                # print(name[-4:])

        # print('*' * 33)
