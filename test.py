import argparse
import numpy as np
import os
import os.path as osp

from dataloaders import make_data_loader
from dataloaders.utils import decode_segmap
from dataloaders.utils import decode_seg_map_sequence
from modeling.deeplab import *
from modeling.sync_batchnorm.replicate import patch_replication_callback
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator
from utils.rail_handler import get_rail_from_mask
from torchvision.utils import save_image
from tqdm import tqdm


class Tester(object):
    def __init__(self, args):
        self.args = args

        # Define DataLoader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define network
        self.model = DeepLab(num_classes=self.nclass,
                             backbone=args.backbone,
                             output_stride=args.out_stride,
                             sync_bn=args.sync_bn,
                             freeze_bn=args.freeze_bn)

        # Define Criterion
        weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)

        # Using cuda
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    # def visualize_image(self, dataset, i, image, target, pred):
    def visualize_image(self, dataset, pred):
        '''
        save_image(image[:4].clone().cpu().data, osp.join(self.args.output_dir, '%d_image.jpg' % i), 2, normalize=True)
        save_image(decode_seg_map_sequence(torch.max(pred[:4], 1)[1].detach().cpu().numpy(),
                                           dataset=dataset), osp.join(self.args.output_dir, '%d_pred.jpg' % i), 2,
                   normalize=False, range=(0, 255))
        '''
        rail, rail_type = get_rail_from_mask(decode_segmap(torch.max(pred[:4], 1)[1].detach().cpu().numpy()[0],
                                                           dataset=dataset))
        '''
        rail = torch.from_numpy(np.array([rail]).transpose([0, 3, 1, 2]))
        save_image(rail, osp.join(self.args.output_dir, '%d_type_is_%s.jpg' % (i, rail_type)), 2, normalize=False,
                   range=(0, 255))
        save_image(decode_seg_map_sequence(torch.squeeze(target[:4], 1).detach().cpu().numpy(),
                                           dataset=dataset), osp.join(self.args.output_dir, '%d_truth.jpg' % i), 2,
                   normalize=False, range=(0, 255))
                   '''

    def test_tensor(self, tensor):
        print(tensor.shape)

    def validation(self):
        if not osp.exists(self.args.output_dir) or not osp.isdir(self.args.output_dir):
            print('%s does not exist or is not a dir' % self.args.output_dir)
            exit(1)
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            self.visualize_image(self.args.dataset, i, image, target, output)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            if i == 19:
                break


def run_model(tester):
    for i, sample in enumerate(tester.val_loader):
        image = sample['image']
        with torch.no_grad():
            output = tester.model(image)


def run_func(tester):
    for i, sample in enumerate(tester.val_loader):
        image = sample['image']
        with torch.no_grad():
            output = tester.model(image)
        tester.visualize_image(tester.args.dataset, output)


def test_fps(args):
    import time
    tester = Tester(args)
    tester.model.eval()
    times = 1
    start = time.time()
    for i in range(times):
        run_model(tester)
    t1 = time.time() - start
    start = time.time()
    for i in range(times):
        run_func(tester)
    t2 = time.time() - start
    t = t2 - t1
    print('FPS = %.5f' % (500 * times / t))


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeepLabV3Plus Testing")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'mobilenet', 'drn'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 16)')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--dataset', type=str, default='coco',
                        choices=['coco'],
                        help='dataset name (default: coco)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='DataLoader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                    training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                    testing (default: auto)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
    False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # output dir
    parser.add_argument('--output_dir', type=str, default='test_result',
                        help='test result output directory')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    test_fps(args)


if __name__ == '__main__':
    main()
