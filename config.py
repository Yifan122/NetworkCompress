class DefaultConfigs(object):
    # 1.string parameters
    train_path = "/mnt/sdb/imagenet/train/"
    test_path = ""
    val_path = "/mnt/sdb/imagenet/val/"

    # 2.numeric parameters
    epochs = 40
    train_batch_size = 16
    val_batch_size = 16
    channel_batch_size = 16

    train_exit = 30  # 3000  # Total number of images for training is train_batch_size*train_exit
    val_exit = 40  # 400 # Total number of images for validation is val_batch_size*val_exit
    channel_exit = 30

    print_fre = 200

    num_workers = 3
    seed = 888
    lr = 0.000001  # 1e-6 for l1-l2 distillation(teacher: Resnet152, Student:ResNet18): 1e-6
    momentum = 0.9
    #device_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    device_ids = [0]
    dev0 = 0

    # temperature and alpha are from distillation
    temperature = 3
    alpha = 0.3
    # due to the attention error is near 1e-3 and kl loss is around 1
    adjusted_beta = 1e3



config = DefaultConfigs()
