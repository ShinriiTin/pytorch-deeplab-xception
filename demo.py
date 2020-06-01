import argparse
import cv2 as cv
import numpy as np
import os
import os.path as osp
import torch
import time

from dataloaders.utils import decode_segmap
from mask_handler import get_rail_from_mask
from modeling.deeplab import *
from modeling.sync_batchnorm.replicate import patch_replication_callback
from utils import demo_transforms as tr
from torchvision import transforms


class Demo(object):
    def __init__(self, args):
        self.args = args

        self.nclass = 21
        # Define network
        self.model = DeepLab(num_classes=self.nclass,
                             backbone=args.backbone,
                             output_stride=args.out_stride,
                             sync_bn=args.sync_bn,
                             freeze_bn=args.freeze_bn)

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

        self.model.eval()

    def predict(self, frame):
        frame = transforms.ToPILImage()(frame).convert('RGB')
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        frame = composed_transforms(frame)
        if self.args.cuda:
            frame = frame.cuda()
        with torch.no_grad():
            pred = self.model(frame)
        rail, rail_type = get_rail_from_mask(torch.max(pred[:4], 1)[1].detach().cpu().numpy()[0].view())
        return decode_segmap(rail, dataset=self.args.dataset), rail_type


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeepLabV3Plus Demo")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
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
    parser.add_argument('--base-size', type=int, default=313,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=313,
                        help='crop image size')
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

    demo = Demo(args)

    output_dir = '/home/shinriitin/GraduationProject/demo_test'

    cap = cv.VideoCapture('/home/shinriitin/GraduationProject/video_data/HX27071_长沙鸿汉_05_A节一端路况_20191109_224501.mp4')
    frames = 0
    cv.namedWindow('InputVideoData')
    cv.moveWindow('InputVideoData', args.crop_size + 150, 50)
    cv.namedWindow('Prediction')
    cv.moveWindow('Prediction', 50, 50)
    t1 = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frames += 1
        rail, rail_type = demo.predict(frame.copy())
        cv.putText(rail, rail_type, (rail.shape[0] // 2, rail.shape[1] // 2), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
                   1, 4)
        cv.imshow('InputVideoData', frame)
        cv.imshow('Prediction', rail)
        if frames % 90 == 0:
            cv.imwrite(osp.join(output_dir, 'InputVideoData_%d.jpg' % frames), frame)
            cv.imwrite(osp.join(output_dir, 'Prediction_%d.jpg' % frames), rail)
        if cv.waitKey(1) == ord('q'):
            break
    t2 = time.time()
    print(frames)
    print(t1)
    print(t2)
    print('FPS: %.5f' % (frames / (t2 - t1)))
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
