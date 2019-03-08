import torch
from utils.accuracy_classifer import accuracy
from utils.func import at_loss, distillation
from utils.averageMeter import AverageMeter
from config import config


def train(train_loader, train_model, baseline_model, criterion, optimizer, epoch=1, exit=1000, print_freq=50,
          device_ids=[0], attention_transfer=False, distillation_knowledge=False, attention_index=None):
    print("starting training")

    if not isinstance(train_model, torch.nn.DataParallel):
        train_model = torch.nn.DataParallel(train_model, device_ids=device_ids)

    train_model.train()
    train_model.cuda()

    if baseline_model is not None:
        if not isinstance(baseline_model, torch.nn.DataParallel):
            baseline_model = torch.nn.DataParallel(baseline_model, device_ids=device_ids)
        baseline_model.eval()
        baseline_model.cuda()

    device = torch.device('cuda', device_ids[0])

    for _ in range(epoch):
        if attention_transfer:
            ATloss = AverageMeter()
        if distillation_knowledge:
            DKloss = AverageMeter()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # register a hook to get the feature map
        if attention_transfer:
            model_feature = [[], [], [], []]
            baseline_model_feature = [[], [], [], []]

            def get_activation(featuremap, index):
                def hook(model, input, output):
                    featuremap[index] = output.detach()

                return hook

            if attention_index is not None:
                model_layer = getattr(train_model.module, 'layer{}'.format(attention_index[0]))
                baseline_model_layer = getattr(baseline_model.module, 'layer{}'.format(attention_index[0]))

                model_layer[attention_index[1]].register_forward_hook(get_activation(model_feature, 0))
                baseline_model_layer[attention_index[1]].register_forward_hook(get_activation(baseline_model_feature, 0))

            else:

                for i in range(1, 5):
                    model_layer = getattr(train_model.module, 'layer{}'.format(i))
                    baseline_model_layer = getattr(baseline_model.module, 'layer{}'.format(i))

                    model_layer[len(model_layer) - 1].register_forward_hook(get_activation(model_feature, i-1))
                    baseline_model_layer[len(baseline_model_layer) - 1].register_forward_hook(
                        get_activation(baseline_model_feature, i-1))

        for i, (input, target) in enumerate(train_loader):

            input = input.cuda()
            target = target.cuda()

            output = train_model(input)
            loss = criterion(output, target)

            if attention_transfer or distillation_knowledge:
                assert baseline_model is not None
                baseline_output = baseline_model(input)

                if distillation_knowledge:
                    dkloss = distillation(output, baseline_output, config.temperature, config.alpha)
                    assert DKloss is not None
                    DKloss.update(dkloss)
                    loss += dkloss

                if attention_transfer:
                    atloss = 0
                    for j in range(len(model_feature)):
                        atloss = at_loss(model_feature[j].to(device), baseline_model_feature[j].to(device))


                    if atloss > 1e-3:
                        scale = int(0.7 / atloss)
                        atloss = scale * atloss
                    else:
                        atloss = atloss * 500

                    assert ATloss is not None
                    ATloss.update(atloss)
                    loss += atloss

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % print_freq == 0:
                print_message = 'Epoch: [{0}][{1}/{2}]\t' \
                                'Top1 {top1.avg:.3f}\t' \
                                'Loss {loss.avg:.4f}\t'.format(
                    epoch, i, len(train_loader), loss=losses, top1=top1)

                if distillation_knowledge:
                    assert DKloss is not None
                    print_message += 'KDLoss {DKloss.avg:.4f}\t'.format(DKloss=DKloss)

                if attention_transfer:
                    assert ATloss is not None
                    print_message += 'ATLoss {ATloss.avg:.4f}\t'.format(ATloss=ATloss)

                print(print_message)
            if i == exit:
                return None

def validate(val_loader, model, criterion, print_fre=50, exit=-1, devices_id=[0], dev0=0):
    # switch to evaluate mode
    model.eval()

    if not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model, device_ids=devices_id)

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    with torch.no_grad():
        with torch.cuda.device(dev0):
            for i, (input, target) in enumerate(val_loader):

                input = input.cuda()
                target = target.cuda()
                model.cuda()

                # compute output
                output = model(input)
                loss = criterion(output, target)

                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.data[0], input.size(0))
                top1.update(prec1[0], input.size(0))
                top5.update(prec5[0], input.size(0))

                # print(top1 / step)

                if i % print_fre == 0:
                    print_message = 'Epoch: [{0}/{1}]\t' \
                                    'Top1 {top1.avg:.3f}\t' \
                                    'Loss {loss.avg:.4f}\t'.format(
                        i, len(val_loader), loss=losses, top1=top1)
                    print(print_message)

                if i == exit:
                    return top1.avg, top5.avg

    return top1.avg, top5.avg
