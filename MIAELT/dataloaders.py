import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
import numpy as np
from PIL import Image
from torchvision import datasets, transforms, utils
import torch.utils.data as data
# from torch.utils.data import Subset, DataLoader, ConcatDataset

class RotationDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.preTransform = transforms.Compose([
                transforms.Resize(64),
                transforms.RandomHorizontalFlip(),
            ])
        self.postTransform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # Output of pretransform should be PIL images
        self.data_dir = root_dir
        self.dataset = datasets.ImageFolder(self.data_dir, self.preTransform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img0, clss = self.dataset[idx]

        rot_class = np.random.randint(4)
        rot_angle = rot_class * 90

        rot_img = img0.rotate(rot_angle)
        if self.postTransform:
            sample = self.postTransform(rot_img)
        else:
            sample = transforms.ToTensor(rot_img)
        return sample, rot_class

def dataset_split(dataset, lengths):
    indices = list(range(sum(lengths)))
    np.random.seed(1)
    np.random.shuffle(indices)
    return [torch.utils.data.Subset(dataset, indices[offset - length:offset]) for offset, length in zip(torch._utils._accumulate(lengths), lengths)]

    return all_data

class custum_CIFAR10(data.Dataset):

    def __init__(self, set_mode, train, config):
        super().__init__()
        self.config = config
        # self.img_size = 32
        self.num_classes = 10
        # self.set_mode = set_mode
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor(), normalize])
        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])
        self.trainset = CIFAR10(root='../data', train=True, transform=self.normalized)
        self.testset = CIFAR10(root='../data', train=False, transform=self.normalized)
        self.data = torch.utils.data.ConcatDataset([self.trainset, self.testset])
        trainsize = config.general.train_target_size
        testsize = config.general.test_target_size
        target_trainset, target_testset, shadow_trainset, shadow_testset, distill_trainset = dataset_split(self.data, [trainsize, testsize, trainsize, testsize, trainsize*2])
        distill_testset = shadow_testset
        self.target_trainset = target_trainset
        self.target_testset = target_testset
        self.shadow_trainset = shadow_trainset
        self.shadow_testset = shadow_testset
        if set_mode == 'target':
            if train:
                #target train size 0:config.general.train_target_size
                self.dataset = target_trainset
            else:
                self.dataset = target_testset
        elif set_mode == 'shadow':
            if train:
                #target train size 0:config.general.train_target_size
                self.dataset = shadow_trainset
            else:
                self.dataset = shadow_testset
        else:
            if train:
                #target train size 0:config.general.train_target_size
                self.dataset = distill_trainset
            else:
                self.dataset = distill_testset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataset[index][0], self.dataset[index][1]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class custum_CIFAR100(data.Dataset):

    def __init__(self, set_mode, train, config):
        super().__init__()
        self.config = config
        # self.img_size = 32
        self.num_classes = 100
        # self.set_mode = set_mode
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor(), normalize])
        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])
        self.trainset = CIFAR100(root='../data', download=True, train=True, transform=self.normalized)
        self.testset = CIFAR100(root='../data', download=True, train=False, transform=self.normalized)
        self.data = torch.utils.data.ConcatDataset([self.trainset, self.testset])
        trainsize = config.general.train_target_size
        testsize = config.general.test_target_size
        target_trainset, target_testset, shadow_trainset, shadow_testset, distill_trainset = dataset_split(self.data, [trainsize, testsize, trainsize, testsize, trainsize*2])
        distill_testset = shadow_testset
        if set_mode == 'target':
            if train:
                #target train size 0:config.general.train_target_size
                self.dataset = target_trainset
            else:
                self.dataset = target_testset
        elif set_mode == 'shadow':
            if train:
                #target train size 0:config.general.train_target_size
                self.dataset = shadow_trainset
            else:
                self.dataset = shadow_testset
        else:
            if train:
                #target train size 0:config.general.train_target_size
                self.dataset = distill_trainset
            else:
                self.dataset = distill_testset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.dataset[index][0], self.dataset[index][1]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class custum_CINIC10(data.Dataset):
    def __init__(self, set_mode, train, config, aug = False):
        self.img_size = 32
        self.num_classes = 10
        self.mean = [0.47889522, 0.47227842, 0.43047404]
        self.std = [0.24205776, 0.23828046, 0.25874835]
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.augmented = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), normalize])

        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset = datasets.ImageFolder(root='./data/cinic-10/train', transform=self.augmented)
        self.aug_testset = datasets.ImageFolder(root='../data/cinic-10/test', transform=self.augmented)
        self.aug_validset = datasets.ImageFolder(root='../data/cinic-10/val', transform=self.augmented)
        self.trainset = datasets.ImageFolder(root='../data/cinic-10/train', transform=self.normalized)
        self.testset = datasets.ImageFolder(root='../data/cinic-10/test', transform=self.normalized)
        self.validset = datasets.ImageFolder(root='../data/cinic-10/val', transform=self.normalized)

        self.aug_dataset = torch.utils.data.ConcatDataset([self.aug_trainset, self.aug_testset, self.aug_validset])
        self.dataset = torch.utils.data.ConcatDataset([self.trainset, self.testset, self.validset])

        trainsize = config.general.train_target_size
        testsize = config.general.test_target_size

        self.aug_target_trainset, self.aug_target_testset, self.aug_shadow_trainset, self.aug_shadow_testset, self.aug_distill_trainset, self.aug_distill_testset = dataset_split(
            self.aug_dataset, [trainsize, testsize, trainsize, testsize, 2*trainsize, testsize])
        self.target_trainset, self.target_testset, self.shadow_trainset, self.shadow_testset, self.distill_trainset, self.distill_testset = dataset_split(
            self.dataset, [trainsize, testsize, trainsize, testsize, 2*trainsize, testsize])

        if set_mode == 'target':
            if aug:
                if train:
                    self.dataset = self.aug_target_trainset
                else:
                    self.dataset = self.aug_target_testset
            else:
                if train:
                    self.dataset = self.target_trainset
                else:
                    self.dataset = self.target_testset
        elif set_mode == 'shadow':
            if aug:
                if train:
                    self.dataset = self.aug_shadow_trainset
                else:
                    self.dataset = self.aug_shadow_testset
            else:
                if train:
                    self.dataset = self.shadow_trainset
                else:
                    self.dataset = self.shadow_testset
        elif 'distill' in set_mode:
            if aug:
                if train:
                    self.dataset = self.aug_distill_trainset
                else:
                    self.dataset = self.aug_distill_testset
            else:
                if train:
                    self.dataset = self.distill_trainset
                else:
                    self.dataset = self.distill_testset

        self.index = range(int(len(self.dataset)))

    def __getitem__(self, idx):
        return self.dataset[idx][0], self.dataset[idx][1], self.index[idx]

    def __len__(self):
        return len(self.index)

def get_data_for_final_eval(models, all_dataloaders, device):
    Y = []
    X = []
    C = []
    for idx_model, model in enumerate(models):
        model.eval()
        #print(all_dataloaders)
        dataloaders = all_dataloaders[idx_model]
        for phase in ['train', 'val']:
            for batch_idx, (data, target) in enumerate(dataloaders[phase]):
                inputs, labels = data.to(device), target.to(device)
                output = model(inputs)
                for out in output.cpu().detach().numpy():
                    X.append(out)
                    if phase == "train":
                        Y.append(1)
                    else:
                        Y.append(0)
                for cla in labels.cpu().detach().numpy():
                    C.append(cla)
    return (np.array(X), np.array(Y), np.array(C))
