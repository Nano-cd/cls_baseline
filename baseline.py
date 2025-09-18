import argparse
import os

import cv2
import matplotlib.pyplot as plt
from torch import GradScaler, nn
from tqdm import tqdm
import numpy as np

from torch.utils.tensorboard import SummaryWriter as sum_writer
import model_hub
import create_dataset
from datasets import build_dataset, build_transform
from model_hub import Alexnet_modify, mobilenet_v2_modify, Resnet18_modify, Shufflenet_v205_modify, squeezenet_modify, \
    Google_net_modify

import torch.autograd
from torch.autograd import Variable
import torch.backends.cudnn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
def parse_config():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_mode", type=str, default='train', help='')
    parser.add_argument("--data_mode", type=str, default='test', help='')
    parser.add_argument('--data_path', type=str, default='dataset_new/train')
    parser.add_argument('--eval_data_path', type=str, default='dataset_new/val')
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--network", type=str, default='alexnet'
                        , help='alexnet')
    parser.add_argument("--train_description1", type=str, default='320',
                        help='train_description')
    parser.add_argument("--seed", type=int, default=19980206)
    parser.add_argument('--ckpt_path', default='./checkpoint', type=str,
                        metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt1', default='Alex_net320-00099.pt',
                        type=str, help='name of the first checkpoint to load')
    parser.add_argument('--tensorboard_path', default='./logs', type=str,
                        metavar='PATH', help='path to checkpoints')

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--number_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--decay_interval", type=int, default=20)
    parser.add_argument("--decay_ratio", type=float, default=0.5)

    return parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Focal_Loss(nn.Module):
    def __init__(self, weight, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # 是tensor数据格式的列表

    def forward(self, preds, labels):
        """
        preds:logist输出值
        labels:标签
        """
        preds = F.softmax(preds, dim=1)
        eps = 1e-7

        target = self.one_hot(preds.size(1), labels).cuda()

        ce = -1 * torch.log(preds + eps) * target
        floss = torch.pow((1 - preds), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)

    def one_hot(self, num, labels):
        one = torch.zeros((labels.size(0), num))
        one[range(labels.size(0)), labels] = 1
        return one



class Trainer(object):
    def __init__(self, config):
        torch.manual_seed(config.seed)
        self.global_step = 0
        self.config = config
        self.data_mode = config.data_mode
        # initialize the data_loader
        self.scaler = GradScaler()
        self.train_dataloader = create_dataset.tuI(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # initialize the model
        if config.network =="alexnet":
            self.model1 = Alexnet_modify.Alex_net(num_classes=3)
        elif config.network =="Mobilenet":
            self.model1 = mobilenet_v2_modify.Mobilenet_v2(num_classes=3)
        elif config.network =="Resnet18":
            self.model1 = Resnet18_modify.Resnet_18(num_classes=3)
        elif config.network =="shuffnet":
            self.model1 = Shufflenet_v205_modify.shufflenet_v2_x0_5(num_classes=3)
        elif config.network =="squeezenet":
            self.model1 = squeezenet_modify.squeezenet1_0(num_classes=3)
        elif config.network =="googlenet":
            self.model1 = Google_net_modify.googlenet(num_classes=3)
        self.model1.to(self.device)
        self.model_name1 = type(self.model1).__name__ + self.config.train_description1
        # initialize the loss function and optimizer
        self.start_epoch = 0
        self.max_epoch = config.max_epoch
        self.loss_fn = Focal_Loss(torch.tensor([0.6,0.3,0.1]).cuda())
        self.ckpt_path = config.ckpt_path
        self.loss_fn.to(self.device)
        self.initial_lr = config.lr
        self.optimizer1 = torch.optim.AdamW(self.model1.parameters(), lr=config.lr)
        self.scheduler1 = torch.optim.lr_scheduler.StepLR(self.optimizer1,
                                                          last_epoch=self.start_epoch - 1,
                                                          step_size=config.decay_interval,
                                                          gamma=config.decay_ratio)
        runs_path = os.path.join(self.config.tensorboard_path, self.model_name1)
        self.logger = sum_writer(str(runs_path))
        if not config.train:
            ckpt1 = os.path.join(str(config.ckpt_path), config.ckpt1)
            self._load_checkpoint(ckpt1=ckpt1)

    def fit(self):
        self.model1.train()
        for epoch in tqdm(range(self.start_epoch, self.max_epoch)):
            self._train_one_epoch(epoch)
            self.scheduler1.step()

        model_name1 = '{}-{:0>5d}.pt'.format(self.model_name1, epoch)
        model_name1 = os.path.join(self.ckpt_path, model_name1)
        self._save_checkpoint({
            'epoch': epoch,
            'state_dict': self.model1.state_dict(),
            'optimizer': self.optimizer1.state_dict(),
        }, model_name1)
    def evl(self):
        y_ = []
        y_pred = []
        self.model1.eval()
        if self.config.data_mode == 'test':
            with torch.no_grad():
                data_pth = 'E:/project_pycharm/TubeDectionP1/dataset_new/TD1_abnormal/images/train/'
                folderlist = os.listdir(data_pth)

                Z1 = []
                Z2 = []
                for i in folderlist:
                    label_path_new = os.path.join(data_pth, i)
                    img = cv2.imread(label_path_new)
                    T = build_transform(False)
                    img = T(img).cuda().unsqueeze(0)
                    R = self.model1(img)
                    Z1.append([torch.argmax(R[0]), R[0]])

                label_pth = 'E:/project_pycharm/TubeDectionP1/dataset/TD1_abnormal/labels/train'
                for filename in os.listdir(label_pth):
                    file_path1 = os.path.join(label_pth, filename)
                    with open(file_path1, 'r') as file:
                        lines = file.readlines()
                        if lines:  # 确保文件不为空
                            first_value_path2 = float(lines[0].strip().split()[0])  # 读取第一个值并转换为浮点数
                            Z2.append([filename.split('.')[0], first_value_path2])
                RS = []
                for j, val in enumerate(Z1):
                    R1 = val[0]
                    R2 = Z2[j][1]
                    if R1 == R2:
                        RS.append([R1, R2, True, Z2[j][0], val[1]])
                    else:
                        RS.append([R1, R2, False, Z2[j][0], val[1]])

                # 将列表转换为CSV格式并保存到文件
                import csv

                # 打开一个新的文件用于写入，如果文件已存在则覆盖
                with open('output.csv', 'w', newline='', encoding='utf-8') as file:
                    # 创建一个csv.writer对象
                    writer = csv.writer(file)

                    # 遍历列表并写入每一行
                    for row in RS:
                        writer.writerow(row)

                print("CSV文件已生成。")
        return y_pred, y_

    def _train_one_epoch(self, epoch):
        for _, data in enumerate(self.train_dataloader):
            x = Variable(data[0])
            y = Variable(data[1])
            # y = torch.reshape(Variable(data[1]),[self.config.batch_size, -1])
            x = x.to(self.device)
            y = y.to(self.device)

            with autocast():
                predict_student, _ = self.model1(x)
                self.loss1 = self.loss_fn(predict_student, y.long())
            self.model1_loss = self.loss1
            self.sum_loss = self.model1_loss

            self.optimizer1.zero_grad()
            self.scaler.scale(self.sum_loss).backward()
            self.scaler.step(self.optimizer1)
            self.scaler.update()

            self.logger.add_scalar(tag='Branch_1!/supervison',
                                   scalar_value=self.loss1.item(),
                                   global_step=self.global_step)
            self.global_step += 1



    def _load_checkpoint(self, ckpt1):
        if os.path.isfile(ckpt1):
            print("[*] loading checkpoint '{}'".format(ckpt1))
            checkpoint = torch.load(ckpt1)
            self.start_epoch = checkpoint['epoch'] + 1
            self.model1.load_state_dict(checkpoint['state_dict'])
            self.optimizer1.load_state_dict(checkpoint['optimizer'])
            if self.initial_lr is not None:
                for param_group in self.optimizer1.param_groups:
                    param_group['initial_lr'] = self.initial_lr
            print("[*] loaded checkpoint '{}' (epoch {})"
                  .format(ckpt1, checkpoint['epoch']))
        else:
            print("[!] no checkpoint found at '{}'".format(ckpt1))

    # save checkpoint
    @staticmethod
    def _save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)


def main(cfg):
    t = Trainer(cfg)
    if cfg.train:
        t.fit()
    else:
        y_pred, y_ = t.evl()
        print(y_pred, y_)


if __name__ == "__main__":
    config = parse_config()
    config.ckpt_path = os.path.join(config.ckpt_path)
    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    print(config.train_description1)
    main(config)
