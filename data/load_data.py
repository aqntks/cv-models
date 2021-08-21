from torchvision import transforms, datasets


def data(data_name):
    # alexnet은 transforms.Resize(256) 이상 사이즈 필수

    if data_name == 'CIFAR_10':
        train_dataset = datasets.CIFAR10(root="data/CIFAR_10",
                                         train=True,
                                         download=True,
                                         transform=transforms.Compose([
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

        test_dataset = datasets.CIFAR10(root="data/CIFAR_10",
                                        train=False,
                                        transform=transforms.Compose([
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    return train_dataset, test_dataset