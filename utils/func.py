import torch.nn.functional as F
import torch


# distillation loss between teacher and student
# Geoffrey Hinton, Oriol Vinyals and Jeff Dean. Distilling the Knowledge in a Neural Network.
def distillation(y, teacher_scores, T, alpha):
    p = F.log_softmax(y / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T ** 2) / y.shape[0]
    # l_ce = F.cross_entropy(y, labels)
    # print("l_kl: %.2f l_ce:%.2f"%(l_kl, l_ce))
    return l_kl * alpha
    #+ l_ce * (1. - alpha)


# Attention Transfer: https://github.com/BayesWatch/pytorch-moonshine
# "Paying More Attention to Attention: Improving the Performance of
#                 Convolutional Neural Networks via Attention Transfer"
# https://github.com/szagoruyko/attention-transfer
def at(x):
    at = F.normalize(x.pow(2).mean(1).view(x.size(0), -1))
    # at = F.normalize(torch.mean(x.pow(2), 1).view(x.size(0), -1))
    return at


def at_loss(g1, g2):
    at_loss = (at(g1) - at(g2)).pow(2).mean()
    return at_loss
