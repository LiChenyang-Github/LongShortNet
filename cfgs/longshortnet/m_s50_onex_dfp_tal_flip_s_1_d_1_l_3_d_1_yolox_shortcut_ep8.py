# encoding: utf-8
import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.data_num_workers = 6
        self.num_classes = 8
        self.input_size = (600, 960)  # (h,w)
        self.random_size = (50, 70)
        self.test_size = (600, 960)
        #
        self.basic_lr_per_img = 0.001 / 64.0

        self.warmup_epochs = 1
        self.max_epoch = 8
        self.no_aug_epochs = 8
        self.eval_interval = 1
        self.train_ann = 'train.json'
        self.val_ann = 'val.json'

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.output_dir = '/data/output'

        self.short_cfg = dict(
                            frame_num=1,
                            delta=1,
                            with_short_cut=False,
                            out_channels=[((96, 192, 384), 1), ],
                        )
        self.long_cfg = dict(
                            frame_num=3,
                            delta=1,
                            with_short_cut=False,
                            include_current_frame=False,
                            out_channels=[((32, 64, 128), 3), ],
                        )
        self.yolox_cfg = dict(
                            merge_form="pure_concat",
                            with_short_cut=True,
                        )


    def get_model(self):
        from exps.model.yolox_longshort_v3 import YOLOXLONGSHORTV3
        from exps.model.dfp_pafpn_long_v3 import DFPPAFPNLONGV3
        from exps.model.dfp_pafpn_short_v3 import DFPPAFPNSHORTV3
        from exps.model.longshort_backbone_neck import BACKBONENECK
        from exps.model.tal_head import TALHead
        import torch.nn as nn

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            long_backbone = (DFPPAFPNLONGV3(self.depth, 
                                            self.width, 
                                            in_channels=in_channels, 
                                            frame_num=self.long_cfg["frame_num"],
                                            with_short_cut=self.long_cfg["with_short_cut"],
                                            out_channels=self.long_cfg["out_channels"])
                            if self.long_cfg["frame_num"] != 0 else None)
            short_backbone = DFPPAFPNSHORTV3(self.depth, 
                                             self.width, 
                                             in_channels=in_channels, 
                                             frame_num=self.short_cfg["frame_num"],
                                             with_short_cut=self.short_cfg["with_short_cut"],
                                             out_channels=self.short_cfg["out_channels"])
            backbone_neck = BACKBONENECK(self.depth, 
                                         self.width, 
                                         in_channels=in_channels)
            head = TALHead(self.num_classes, self.width, in_channels=in_channels, gamma=1.0,
                             ignore_thr=0.4, ignore_value=1.7)
            self.model = YOLOXLONGSHORTV3(long_backbone, 
                                          short_backbone, 
                                          backbone_neck,
                                          head, 
                                          merge_form=self.yolox_cfg["merge_form"], 
                                          in_channels=in_channels, 
                                          width=self.width,
                                          with_short_cut=self.yolox_cfg["with_short_cut"])

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, local_rank=0, cache_img=False):
        from exps.dataset.longshort.tal_flip_long_short_argoversedataset import LONGSHORT_ARGOVERSEDataset
        from exps.data.tal_flip_mosaicdetection import LongShortMosaicDetection
        from exps.data.data_augment_flip import LongShortTrainTransform
        from yolox.data import (
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            worker_init_reset_seed,
        )

        dataset = LONGSHORT_ARGOVERSEDataset(
            data_dir='/data',
            json_file=self.train_ann,
            name='train',
            img_size=self.input_size,
            preproc=LongShortTrainTransform(max_labels=50, 
                                            hsv=False, 
                                            flip=True, 
                                            short_frame_num=self.short_cfg["frame_num"], 
                                            long_frame_num=self.long_cfg["frame_num"]),
            cache=cache_img,
            short_cfg=self.short_cfg,
            long_cfg=self.long_cfg,
        )

        dataset = LongShortMosaicDetection(dataset,
                                          mosaic=not no_aug,
                                          img_size=self.input_size,
                                          preproc=LongShortTrainTransform(max_labels=120, 
                                                                          hsv=False, 
                                                                          flip=True, 
                                                                          short_frame_num=self.short_cfg["frame_num"], 
                                                                          long_frame_num=self.long_cfg["frame_num"]),
                                          degrees=self.degrees,
                                          translate=self.translate,
                                          scale=self.mosaic_scale,
                                          shear=self.shear,
                                          perspective=0.0,
                                          enable_mixup=self.enable_mixup,
                                          mosaic_prob=self.mosaic_prob,
                                          mixup_prob=self.mixup_prob,
                                        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug)

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        # from exps.dataset.tal_flip_one_future_argoversedataset import ONE_ARGOVERSEDataset
        # from exps.data.data_augment_flip import DoubleValTransform
        from exps.dataset.longshort.tal_flip_long_short_argoversedataset import LONGSHORT_ARGOVERSEDataset
        from exps.data.data_augment_flip import LongShortValTransform

        valdataset = LONGSHORT_ARGOVERSEDataset(
            data_dir='/data',
            json_file='val.json',
            name='val',
            img_size=self.test_size,
            preproc=LongShortValTransform(short_frame_num=self.short_cfg["frame_num"],
                                          long_frame_num=self.long_cfg["frame_num"]),
            short_cfg=self.short_cfg,
            long_cfg=self.long_cfg,
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "sampler": sampler}
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        import random
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            if epoch >= self.max_epoch - 1:
                size = self.input_size
            else:
                size_factor = self.input_size[0] * 1.0 / self.input_size[1]
                size = random.randint(*self.random_size)
                size = (16 * int(size * size_factor), int(16 * size))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size
    

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs[0] = nn.functional.interpolate(
                inputs[0], size=tsize, mode="bilinear", align_corners=False
            )
            inputs[1] = nn.functional.interpolate(
                inputs[1], size=tsize, mode="bilinear", align_corners=False
            ) if inputs[1].ndim == 4 else inputs[1] # inputs[1].ndim != 4 为不使用long支路的情况
            targets[0][..., 1::2] = targets[0][..., 1::2] * scale_x
            targets[0][..., 2::2] = targets[0][..., 2::2] * scale_y
            targets[1][..., 1::2] = targets[1][..., 1::2] * scale_x
            targets[1][..., 2::2] = targets[1][..., 2::2] * scale_y
        return inputs, targets

    def get_evaluator(self, batch_size, is_distributed, testdev=False):
        # from exps.evaluators.onex_stream_evaluator import ONEX_COCOEvaluator
        from exps.evaluators.longshort_onex_stream_evaluator import LONGSHORT_ONEX_COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev)
        evaluator = LONGSHORT_ONEX_COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator


    def get_trainer(self, args):
        from exps.train_utils.longshort_trainer import Trainer
        trainer = Trainer(self, args)
        # NOTE: trainer shouldn't be an attribute of exp object
        return trainer

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)


