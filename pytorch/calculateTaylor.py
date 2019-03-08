import torch
import torch.nn as nn
import torch.optim as optim
from config import config
from utils.averageMeter import AverageMeter
from utils.accuracy_classifer import accuracy
from pytorch.data_loader import get_data_loader


def calculateTaylor(pruner, epoch = 1, print_freq = 200):
    dataloader = get_data_loader(config.train_path, config.val_batch_size, config.num_workers, type='train')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(pruner.model.parameters(), config.lr,
                          momentum=config.momentum)

    if not isinstance(pruner.model, torch.nn.DataParallel):
        train_model = torch.nn.DataParallel(pruner.model, device_ids=config.device_ids)

    pruner.model.train()
    pruner.model.cuda()

    for _ in range(epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for i, (input, target) in enumerate(dataloader):
            input = input.cuda()
            target = target.cuda()

            output = pruner.forward(input)
            loss = criterion(output, target)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()

            if i % print_freq == 0:
                print_message = 'Epoch: [{0}][{1}/{2}]\t' \
                                'Top1 {top1.avg:.3f}\t' \
                                'Loss {loss.avg:.4f}\t'.format(
                    epoch, i, len(dataloader), loss=losses, top1=top1)

                print(print_message)
            if i == 20000:
                return None