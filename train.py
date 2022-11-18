from torch.autograd import Variable
import os, argparse
from utils.x_change_model import *
from dataloader.change_dataloader import get_loader
from visualize.process_bar import process_bar
from utils.lr_decay import adjust_lr
from backbone.resnet import *
import time


def train(train_loader, model, optimizer, epoch):
    model.train()
    total_step = len(train_loader)
    loss_sum, avg_loss, process_step, cont_loss_sum = 0, 0, 0, 0

    if opt.mixed_precision_training:
        scaler = torch.cuda.amp.GradScaler()

    global_start = time.perf_counter()
    for i, pack in enumerate(train_loader, start=1):
        start = time.perf_counter()

        optimizer.zero_grad()
        before, after, gts = pack

        '''Data process'''
        before = Variable(before)
        after = Variable(after)
        gts = Variable(gts)

        before = before.cuda()
        after = after.cuda()
        gts = gts.cuda()

        '''Calculate loss and backward'''
        if opt.mixed_precision_training:
            with torch.cuda.amp.autocast():
                '''Get predict mask'''
                dets = model(before, after)
            loss = BCE(dets, gts)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            '''Get predict mask'''
            dets = model(before, after)
            loss = BCE(dets, gts)
            loss.backward()
            optimizer.step()
        '''Training progress visualize'''
        loss_sum += loss.cpu().item()
        step_loss = loss.cpu().item()
        avg_loss = loss_sum / i
        '''Process bar'''
        process_step += 100 / total_step

        '''calc run time'''
        end = time.perf_counter()
        run_time = end - start
        last_time = "%d:%d" % (run_time * (total_step - i) // 60, run_time * (total_step - i) % 60)
        past_time = "%d:%d" % ((end - global_start) // 60, (end - global_start) % 60)
        time_str = "[{}/{}]".format(past_time, last_time)

        process_bar(process_step, epoch, avg_loss, optimizer.param_groups[0]['lr'], step_loss, time_str)
        '''Training progress visualize'''

    '''Training progress visualize'''
    print(' Epoch ' + str(epoch) + ' train loss: \033[34m%.4f\033[0m' % avg_loss)

    '''Save checkpoints'''
    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % opt.save_feq == 0:
        torch.save(model.state_dict(), save_path + 'Epoch_%d_loss_%.4f_CD.pth' % (epoch, avg_loss))


def val(val_loader, model, epoch):
    model.eval()
    total_step = len(val_loader)
    loss_sum, process_step = 0, 0
    TP, TN, FP, FN = 0, 0, 0, 0
    with torch.no_grad():
        for i, pack in enumerate(val_loader, start=1):
            before, after, gts = pack
            '''Data process'''
            before = Variable(before)
            after = Variable(after)
            gts = Variable(gts)

            before = before.cuda()
            after = after.cuda()
            gts = gts.cuda()

            '''Get predict mask'''
            dets = model(before, after)

            '''Calculate loss and backward'''
            loss = BCE(dets, gts)

            '''evaluation'''
            dets = dets.sigmoid()
            dets = torch.where(dets <= 0.5, 0., 1.)
            dets = dets + 2  # (2 , 3)
            gts = gts + 1  # (1 , 2)
            iou_map = gts * dets

            TP += len((iou_map == 6).nonzero())  # change predict to change -> 2 * 3 -> TP
            FN += len((iou_map == 4).nonzero())  # change predict to unchange -> 2 * 2 -> FN
            FP += len((iou_map == 3).nonzero())  # unchange predict to change -> 1 * 3 -> FP
            TN += len((iou_map == 2).nonzero())  # unchange predict to unchange -> 1 * 2 -> FN

            '''Val progress visualize'''
            loss_sum += loss.cpu().item()
            avg_loss = loss_sum / i
            '''Process bar'''
            process_step += 100 / total_step
            process_bar(process_step, epoch, avg_loss, training=False)
            '''Training progress visualize'''

    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    iou = TP / (TP + FP + FN + 1e-10)
    F1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    '''Val progress visualize'''
    print(' Epoch ' + str(epoch) + '   val loss: %.4f pre: %.4f recall: %.4f iou:%.4f,F1:%.4f' % (
        avg_loss, precision, recall, iou, F1))

    '''Save checkpoints'''
    if F1 > opt.best_f1:
        opt.best_f1 = F1
        save_path = opt.save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # torch.save(model.state_dict(), save_path + 'Best_CD_{}.pth'.format(F1))
        torch.save(model.state_dict(), save_path + 'Best_CD.pth')

    with open("log.txt", "a") as file:
        file.write(
            "Epoch %d test loss %.4f Precision %.4f Recall %.4f iou %.4f F1 %.4f\n" % (
                epoch, avg_loss, precision, recall, iou, F1))


if __name__ == '__main__':
    '''Train parameter'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=500, help='epoch number')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='set weight decay')
    parser.add_argument('--mixed_precision_training', action="store_false", help='enable mixed precision training')
    parser.add_argument('--train_batchsize', type=int, default=4, help='training batch size')
    parser.add_argument('--val_batchsize', type=int, default=4, help='val batch size')
    parser.add_argument('--trainsize', type=int, default=1024, help='training dataset size')
    parser.add_argument('--best_f1', type=float, default=0.0, help='f1 record')
    parser.add_argument('--val_epoch', type=int, default=1, help='every n epochs do evaluation')
    parser.add_argument('--decay_epoch', type=int, default=1, help='every n epochs decay learning rate')
    parser.add_argument('--backbone', default='18', help='set backbone in {18, 34, 50, 101}')
    parser.add_argument('--operation_type', default='cat', help='set operation in {cat, sub}')
    parser.add_argument('--enable_x_cross', action="store_true", help='set with cross attention or not')
    parser.add_argument('--root', default='data/LEVIR-CD/', help='root dir')
    parser.add_argument('--save_path', default='checkpoints/run/', help='checkpoint save dir')
    parser.add_argument('--save_feq', default=10, help='checkpoint save dir')
    parser.add_argument('--checkpoint_path',
                        default='checkpoints/saves/res18_cross1_9193.pth',
                        help='checkpoint path for resume')
    parser.add_argument('--resume', action="store_true",
                        help='resume checkpoint (set to False will load model pretrained on imagenet)')
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    '''Build models'''
    ''' backbone in {18, 34, 50, 101}  set backbone volume
        x_cross in {True, False} set with cross attention or not
        operation_type in {cat, sub} set difference operation '''
    model = Resnet_CD(backbone=opt.backbone, x_cross=opt.enable_x_cross, operation_type=opt.operation_type)
    print(model)
    model.cuda()
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, opt.lr, eps=1e-3, weight_decay=opt.weight_decay)

    '''Resume model'''
    model_path = opt.checkpoint_path
    if opt.resume:
        print('Loading weights ' + model_path.split('/')[-1] + ' into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print('Finished!')
    else:
        pretrained_path = 'checkpoints/saves/resnet{}.pth'.format(opt.backbone)
        if os.path.exists(pretrained_path):
            backbone_dict = torch.load(pretrained_path)
            model.backbone.load_state_dict(backbone_dict, strict=False)
            print('Load model pretrained on imagenet, Finished!')

    '''Load training data'''
    root = opt.root
    train_before, train_after, train_gt = root + 'train/A/', root + 'train/B/', root + 'train/label/'
    val_before, val_after, val_gt = root + 'test/A/', root + 'test/B/', root + 'test/label/'
    train_loader = get_loader(train_before, train_after, train_gt, batchsize=opt.train_batchsize,
                              trainsize=opt.trainsize,
                              num_workers=4)
    val_loader = get_loader(val_before, val_after, val_gt, batchsize=opt.val_batchsize, trainsize=opt.trainsize,
                            advance=False)

    '''Loss function'''
    BCE = torch.nn.BCEWithLogitsLoss()

    '''Start Training'''
    print('Learning Rate: {} Total Epoch: {}'.format(opt.lr, opt.epoch))
    print('Train with x-cross state: {}, Lets go! '.format(opt.enable_x_cross))
    for epoch in range(1, opt.epoch + 1):
        train(train_loader, model, optimizer, epoch)
        if epoch % opt.val_epoch == 0:
            val(val_loader, model, epoch)
        if epoch % opt.decay_epoch == 0:
            adjust_lr(optimizer, opt.lr, epoch, opt.epoch, power=2)
