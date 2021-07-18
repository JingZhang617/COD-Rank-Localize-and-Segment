import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from datetime import datetime
from torch.optim import lr_scheduler
from model.ResNet_models import Generator
from data import get_loader
from utils import adjust_lr, AvgMeter
from scipy import misc
import cv2
import torchvision.transforms as transforms
from utils import l2_regularisation



parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=50, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=2.5e-5, help='learning rate for generator')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=480, help='training dataset size')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of saliency feat')
opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
generator = Generator(channel=opt.feat_channel)
generator.cuda()
generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)


image_root = './dataset/train/Imgs/'
gt_root = './dataset/train/GT/'
fix_root = './dataset/train/Fix/'

train_loader = get_loader(image_root, gt_root, fix_root,batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
size_rates = [0.75,1,1.25]  # multi-scale training

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_cod1(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_cod1.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_cod2(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_cod2.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_fix(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_fix.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

def visualize_fix_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_fix_gt.png'.format(kk)
        cv2.imwrite(save_path + name, pred_edge_kk)

for epoch in range(1, (opt.epoch+1)):
    # scheduler.step()
    generator.train()
    loss_record = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            images, gts, fixs = pack
            images = Variable(images)
            gts = Variable(gts)
            fixs = Variable(fixs)
            images = images.cuda()
            gts = gts.cuda()
            fixs = fixs.cuda()
            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                fixs = F.upsample(fixs, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            fix_pred,cod_pred1,cod_pred2 = generator.forward(images)
            fix_loss = mse_loss(torch.sigmoid(fix_pred),fixs)
            cod_loss1 = structure_loss(cod_pred1, gts)
            cod_loss2 = structure_loss(cod_pred2, gts)
            final_loss = fix_loss + cod_loss1 + cod_loss2
            final_loss.backward()
            generator_optimizer.step()

            visualize_cod1(torch.sigmoid(cod_pred1))
            visualize_cod2(torch.sigmoid(cod_pred2))
            visualize_fix(torch.sigmoid(fix_pred))
            visualize_fix_gt(fixs)
            visualize_gt(gts)

            if rate == 1:
                loss_record.update(final_loss.data, opt.batchsize)

        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Gen Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))

    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

    save_path = 'models/Resnet/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % opt.epoch == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
