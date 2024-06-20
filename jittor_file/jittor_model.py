import jittor as jt
from jittor.models.resnet import *
from jittor import init
from jittor import nn
from jittor_pd import GL_CLASSES, GL_NUMBBOX, GL_NUMGRID
from util import calculate_iou

class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        resnet = Resnet34(pretrained=True)
        resnet_out_channel = resnet.fc.in_features
        self.resnet = nn.Sequential(*list(resnet.children())[:(- 2)])
        self.Conv_layers = nn.Sequential(
            nn.Conv(resnet_out_channel, 1024, 3, padding=1),
            nn.BatchNorm(1024),
            nn.LeakyReLU(),
            nn.Conv(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm(1024),
            nn.LeakyReLU(),
            nn.Conv(1024, 1024, 3, padding=1),
            nn.BatchNorm(1024),
            nn.LeakyReLU(),
            nn.Conv(1024, 1024, 3, padding=1),
            nn.BatchNorm(1024),
            nn.LeakyReLU())
        self.Conn_layers = nn.Sequential(
            nn.Linear(((GL_NUMGRID * GL_NUMGRID) * 1024), 4096),
            nn.LeakyReLU(),
            nn.Linear(4096, ((GL_NUMGRID * GL_NUMGRID) * ((5 * GL_NUMBBOX) + len(GL_CLASSES)))),
            nn.Sigmoid()
        )

    def execute(self, inputs):
        x = self.resnet(inputs)
        x = self.Conv_layers(x)
        x = x.view((x.shape[0], (- 1)))
        x = self.Conn_layers(x)
        self.pred = x.reshape(((- 1), ((5 * GL_NUMBBOX) + len(GL_CLASSES)), GL_NUMGRID, GL_NUMGRID))
        return self.pred

    def calculate_loss(self, labels):
        self.pred = self.pred.double()
        labels = labels.double()
        (num_gridx, num_gridy) = (GL_NUMGRID, GL_NUMGRID)
        noobj_confi_loss = 0.0
        coor_loss = 0.0
        obj_confi_loss = 0.0
        class_loss = 0.0
        n_batch = labels.shape[0]
        for i in range(n_batch):
            for n in range(num_gridx):
                for m in range(num_gridy):
                    if (labels[(i, 4, m, n)] == 1):
                        bbox1_pred_xyxy = ((((self.pred[(i, 0, m, n)] + n) / num_gridx) - (self.pred[(i, 2, m, n)] / 2)), (((self.pred[(i, 1, m, n)] + m) / num_gridy) - (self.pred[(i, 3, m, n)] / 2)), (((self.pred[(i, 0, m, n)] + n) / num_gridx) + (self.pred[(i, 2, m, n)] / 2)), (((self.pred[(i, 1, m, n)] + m) / num_gridy) + (self.pred[(i, 3, m, n)] / 2)))
                        bbox2_pred_xyxy = ((((self.pred[(i, 5, m, n)] + n) / num_gridx) - (self.pred[(i, 7, m, n)] / 2)), (((self.pred[(i, 6, m, n)] + m) / num_gridy) - (self.pred[(i, 8, m, n)] / 2)), (((self.pred[(i, 5, m, n)] + n) / num_gridx) + (self.pred[(i, 7, m, n)] / 2)), (((self.pred[(i, 6, m, n)] + m) / num_gridy) + (self.pred[(i, 8, m, n)] / 2)))
                        bbox_gt_xyxy = ((((labels[(i, 0, m, n)] + n) / num_gridx) - (labels[(i, 2, m, n)] / 2)), (((labels[(i, 1, m, n)] + m) / num_gridy) - (labels[(i, 3, m, n)] / 2)), (((labels[(i, 0, m, n)] + n) / num_gridx) + (labels[(i, 2, m, n)] / 2)), (((labels[(i, 1, m, n)] + m) / num_gridy) + (labels[(i, 3, m, n)] / 2)))
                        iou1 = calculate_iou(bbox1_pred_xyxy, bbox_gt_xyxy)
                        iou2 = calculate_iou(bbox2_pred_xyxy, bbox_gt_xyxy)
                        if (iou1 >= iou2):
                            coor_loss = (coor_loss + (5 * (jt.Var.sum(((self.pred[i, 0:2, m, n] - labels[i, 0:2, m, n]) ** 2)) + jt.Var.sum(((self.pred[i, 2:4, m, n].sqrt() - labels[i, 2:4, m, n].sqrt()) ** 2)))))
                            obj_confi_loss = (obj_confi_loss + ((self.pred[(i, 4, m, n)] - iou1) ** 2))
                            noobj_confi_loss = (noobj_confi_loss + (0.5 * ((self.pred[(i, 9, m, n)] - iou2) ** 2)))
                        else:
                            coor_loss = (coor_loss + (5 * (jt.Var.sum(((self.pred[i, 5:7, m, n] - labels[i, 5:7, m, n]) ** 2)) + jt.Var.sum(((self.pred[i, 7:9, m, n].sqrt() - labels[i, 7:9, m, n].sqrt()) ** 2)))))
                            obj_confi_loss = (obj_confi_loss + ((self.pred[(i, 9, m, n)] - iou2) ** 2))
                            noobj_confi_loss = (noobj_confi_loss + (0.5 * ((self.pred[(i, 4, m, n)] - iou1) ** 2)))
                        class_loss = (class_loss + jt.Var.sum(((self.pred[i, 10:, m, n] - labels[i, 10:, m, n]) ** 2)))
                    else:
                        noobj_confi_loss = (noobj_confi_loss + (0.5 * jt.Var.sum((self.pred[(i, [4, 9], m, n)] ** 2))))
        loss = (((coor_loss + obj_confi_loss) + noobj_confi_loss) + class_loss)
        return (loss / n_batch)

    def calculate_metric(self, preds, labels):
        preds = preds.double()
        labels = labels[:, :(self.n_points * 2)]
        l2_distance = jt.Var.mean(jt.Var.sum(((preds - labels) ** 2), dim=1))
        return l2_distance

if __name__ == '__main__':
    # 自定义输入张量，验证网络可以正常跑通，并计算loss，调试用
    x = jt.zeros(5,3,448,448)
    net = MyNet()
    a = net(x)
    labels = jt.zeros(5, 30, 7, 7)
    loss = net.calculate_loss(labels)
    print(loss)
    print(a.shape)