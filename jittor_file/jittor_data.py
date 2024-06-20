import jittor as jt
from jittor.dataset import Dataset, DataLoader
from jittor import init
from jittor import nn
import numpy as np
import os
import random
from PIL import Image

class MyDataset(Dataset):

    def __init__(self, dataset_dir, seed=None, mode='train', train_val_ratio=0.9, trans=None):
        super().__init__()
        if (seed is None):
            seed = random.randint(0, 65536)
        random.seed(seed)
        self.dataset_dir = dataset_dir
        self.mode = mode
        if (mode == 'val'):
            mode = 'train'
        img_list_txt = os.path.join(dataset_dir, (mode + '.txt'))
        label_csv = os.path.join(dataset_dir, (mode + '.csv'))
        self.img_list = []
        self.label = np.loadtxt(label_csv)
        with open(img_list_txt, 'r') as f:
            for line in f.readlines():
                self.img_list.append(line.strip())
        self.num_all_data = len(self.img_list)
        all_ids = list(range(self.num_all_data))
        num_train = int((train_val_ratio * self.num_all_data))
        if (self.mode == 'train'):
            self.use_ids = all_ids[:num_train]
        elif (self.mode == 'val'):
            self.use_ids = all_ids[num_train:]
        else:
            self.use_ids = all_ids
        self.trans = trans

    def __len__(self):
        return len(self.use_ids)

    def __getitem__(self, item):
        id = self.use_ids[item]
        label = self.label[id, :]
        img_path = self.img_list[id]
        img = Image.open(img_path)
        if (self.trans is None):
            trans = jt.transform.Compose([jt.transform.ToTensor()])
        else:
            trans = self.trans
        img = trans(img)
        return (img, label)




if (__name__ == '__main__'):
    dataset_dir = '../dataset/cby_yawn_forYolov1'
    dataset = MyDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=1)
    for i in enumerate(dataloader):
        input('press enter to continue')

