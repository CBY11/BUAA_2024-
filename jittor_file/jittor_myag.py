import jittor as jt
from jittor import init
from jittor import nn
import argparse

class Args(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def set_train_args(self):
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--lr', type=float, default=1e-02, help='learning rate')
        self.parser.add_argument('--weight_decay', type=float, default=1e-08)
        self.parser.add_argument('--epoch', type=int, default=10, help='number of end epoch')
        self.parser.add_argument('--start_epoch', type=int, default=0, help='number of start epoch')
        self.parser.add_argument('--use_GPU', action='store_true', help='identify whether to use gpu')
        self.parser.add_argument('--GPU_id', type=int, default=None, help='device id')
        self.parser.add_argument('--dataset_dir', type=str, default='F:\\Develop\\Python_File\\DownLoad\\YOLOv1-from-scratch\\dataset\\cby_yawn_forYolov1')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
        self.parser.add_argument('--print_freq', type=int, default=20, help='print training information frequency (per n iteration)')
        self.parser.add_argument('--save_freq', type=int, default=1, help='save model frequency (per n epoch)')
        self.parser.add_argument('--num_workers', type=int, default=1, help='use n threads to read data')
        self.parser.add_argument('--pretrain', type=str, default=None, help='pretrain model path')
        self.parser.add_argument('--random_seed', type=int, default=0, help='random seed for split dataset')
        self.opts = self.parser.parse_args()
        if False:
            self.opts.use_GPU = True
            self.opts.GPU_id = 1
            print(('use GPU %d to train.' % self.opts.GPU_id))
        else:
            print('use CPU to train.')

    def set_test_args(self):
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--use_GPU', action='store_true', help='identify whether to use gpu')
        self.parser.add_argument('--GPU_id', type=int, default=None, help='device id')
        self.parser.add_argument('--dataset_dir', type=str, default='F:\\Develop\\Python_File\\DownLoad\\YOLOv1-from-scratch\\dataset\\cby_yawn_forYolov1\\img')
        self.parser.add_argument('--weight_path', type=str, default='F:\\Develop\\Python_File\\DownLoad\\YOLOv1-from-scratch\\checkpoints\\epoch7.pkl', help='load path for model weight')
        self.opts = self.parser.parse_args()
        if False:
            self.opts.use_GPU = True
            self.opts.GPU_id = 1
            print(('use GPU %d to train.' % self.opts.GPU_id))
        else:
            print('use CPU to train.')

    def get_opts(self):
        return self.opts

