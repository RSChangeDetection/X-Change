import os
import numpy as np
import argparse

from utils.x_change_model import *
from dataloader.change_dataloader import test_dataset
from tqdm import tqdm
from PIL import Image


def save_mask(res, save_path, name):
    res = torch.where(res <= 0.5, 0., 255.)
    img = np.uint8(res.squeeze().data.cpu().numpy())
    img = Image.fromarray(img)
    img.save(save_path + name)


def save_iou_map(iou_map, size, path, index):
    img = torch.zeros([size, size, 3])
    img[(iou_map == 4).nonzero()[:, 0], (iou_map == 4).nonzero()[:, 1], 0] = 200.
    img[(iou_map == 6).nonzero()[:, 0], (iou_map == 6).nonzero()[:, 1], 1] = 200.
    img[(iou_map == 3).nonzero()[:, 0], (iou_map == 3).nonzero()[:, 1], 2] = 200.
    img = np.uint8(img.data.cpu().numpy())
    img = Image.fromarray(img)
    img.save(path + index)


def test_and_eval():
    '''Test parameter'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=1024, help='testing size')
    parser.add_argument('--save_result', action="store_true", help='test root dir')
    parser.add_argument('--save_iou_map', action="store_true", help='test root dir')
    parser.add_argument('--save_result_path', default='result/predict/', help='test root dir')
    parser.add_argument('--save_iou_map_path', default='result/iou_map/', help='test root dir')
    parser.add_argument('--backbone', default='18', help='set backbone in {18, 34, 50, 101}')
    parser.add_argument('--operation_type', default='cat', help='set operation in {cat, sub}')
    parser.add_argument('--enable_x_cross', default=True, help='set with cross attention or not')
    parser.add_argument('--checkpoint_path', default='checkpoints/saves/res18_best.pth',
                        help='checkpoint path for test')
    parser.add_argument('--root', default='data/LEVIR-CD/test/', help='test root dir')
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    '''Bulid model and load checkpoint'''
    model = Resnet_CD(backbone=opt.backbone, x_cross=opt.enable_x_cross, operation_type=opt.operation_type)
    print(model)
    print('Test with x-cross state: {}'.format(opt.enable_x_cross))
    model.load_state_dict(torch.load('checkpoints/saves/res18_cross1_9193.pth'), strict=False)
    model.cuda()
    model.eval()

    # model = load_state_dict_from_zero_checkpoint(model, '/root/change/checkpoints/run/Best_CD_0.9238543662475396.pth')

    if not os.path.exists(opt.save_result_path) and opt.save_result:
        os.makedirs(opt.save_result_path)
    if not os.path.exists(opt.save_iou_map_path) and opt.save_iou_map:
        os.makedirs(opt.save_iou_map_path)

    '''Load test data'''
    image_before, image_after, gt_root = opt.root + 'A/', opt.root + 'B/', opt.root + 'label/'
    test_loader = test_dataset(image_before, image_after, gt_root, opt.testsize)

    '''Test'''
    with torch.no_grad():
        TP, TN, FP, FN = 0, 0, 0, 0
        for _ in tqdm(range(test_loader.size)):
            before, after, gt, name = test_loader.load_data()
            before = before.cuda()
            after = after.cuda()
            gt = gt.cuda()

            results = model(before, after)
            results = results.sigmoid()
            mask = torch.where(results <= 0.5, 0., 1.)
            mask = mask + 2  # (2, 3)
            gt = gt + 1  # (1, 2)
            iou_map = gt * mask

            TP += len((iou_map == 6).nonzero())  # change predict to change -> 2 * 3 -> TP
            FN += len((iou_map == 4).nonzero())  # change predict to unchange -> 2 * 2 -> FN
            FP += len((iou_map == 3).nonzero())  # unchange predict to change -> 1 * 3 -> FP
            TN += len((iou_map == 2).nonzero())  # unchange predict to unchange -> 1 * 2 -> TN

            if opt.save_result:
                save_mask(results, opt.save_result_path, name)
            if opt.save_iou_map:
                save_iou_map(iou_map.squeeze(), opt.testsize, opt.save_iou_map_path, name)

    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    iou = TP / (TP + FP + FN + 1e-10)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    print('precision: %.2f %%' % (precision * 100))
    print('   recall: %.2f %%' % (recall * 100))
    print('      iou: %.2f %%' % (iou * 100))
    print('       F1: %.2f %%' % (F1    * 100))


if __name__ == '__main__':
    test_and_eval()
