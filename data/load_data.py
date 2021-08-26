from torchvision import transforms, datasets


def data(data_name, image_size):
    if image_size == -1:
        trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        trans = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # alexnet은 transforms.Resize(256) 이상 사이즈 필수

    if data_name == 'CIFAR_10':
        train_dataset = datasets.CIFAR10(root="data/CIFAR_10",
                                         train=True,
                                         download=True,
                                         transform=trans)

        test_dataset = datasets.CIFAR10(root="data/CIFAR_10",
                                        train=False,
                                        transform=trans)

    # if data_name == 'CIFAR_100':
    #     train_dataset = datasets.CIFAR100(root="data/CIFAR_100",
    #                                      train=True,
    #                                      download=True,
    #                                      transform=transforms.Compose([
    #                                          transforms.RandomHorizontalFlip(),
    #                                          transforms.ToTensor(),
    #                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    #
    #     test_dataset = datasets.CIFAR100(root="data/CIFAR_100",
    #                                     train=False,
    #                                     transform=transforms.Compose([
    #                                         transforms.RandomHorizontalFlip(),
    #                                         transforms.ToTensor(),
    #                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    if data_name == 'SVHN':
        train_dataset = datasets.SVHN(root="data/SVHN",
                                         split='train',
                                         download=True,
                                         transform=trans)

        test_dataset = datasets.SVHN(root="data/SVHN",
                                        split='test',
                                        download=True,
                                        transform=trans)

    # if data_name == 'CelebA':
    #     train_dataset = datasets.CelebA(root="data/CelebA",
    #                                      split='train-standard',
    #                                      download=True,
    #                                      transform=transforms.Compose([
    #                                          transforms.RandomHorizontalFlip(),
    #                                          transforms.ToTensor(),
    #                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    #
    #     test_dataset = datasets.CelebA(root="data/CelebA",
    #                                     split='val',
    #                                     download=True,
    #                                     transform=transforms.Compose([
    #                                         transforms.RandomHorizontalFlip(),
    #                                         transforms.ToTensor(),
    #                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

    if data_name == 'Places365':
        train_dataset = datasets.Places365(root="data/Places365",
                                         split='train-standard',
                                         download=True,
                                         transform=trans)

        test_dataset = datasets.Places365(root="data/Places365",
                                        split='val',
                                        download=True,
                                        transform=trans)

    if data_name == 'STL10':
        train_dataset = datasets.STL10(root="data/STL10",
                                         split='train',
                                         download=True,
                                         transform=trans)

        test_dataset = datasets.STL10(root="data/STL10",
                                        split='test',
                                        download=True,
                                        transform=trans)

    return train_dataset, test_dataset