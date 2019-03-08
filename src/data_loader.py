import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_data_loader(data_path, batch_size, num_worker, type='val'):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if type == 'val':
        dataset = datasets.ImageFolder(
            data_path,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
    elif type == 'train':
        dataset = datasets.ImageFolder(
            data_path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                              num_workers=num_worker, pin_memory=True)

    return data_loader
