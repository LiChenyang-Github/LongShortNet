# Modified based on tal_flip_img_one_two_future_argoversedataset.py. Support onex and twox(image and annotations) training in one dataset.
import cv2
import copy
import numpy as np
from enum import Enum
# import json
# import time
# from PIL import Image
from pycocotools.coco import COCO
# from collections import defaultdict

# import io
import os

# from yolox.data.dataloading import get_yolox_datadir
from yolox.data.datasets.datasets_wrapper import Dataset

# from loguru import logger


class SpeedType(Enum):
    ONEX = "onex"
    TWOX = "twox"


class IMG_ANNO_ONE_TWO_ARGOVERSEDataset(Dataset):
    """
    COCO dataset class. 
    """
    def __init__(self, data_dir='/data/Datasets/', json_file='train.json',
                 name='train', img_size=(416,416), preproc=None, cache=False, 
                 speed_prob={SpeedType.ONEX.value: 0.5, SpeedType.TWOX.value: 0.5}, 
                 speed_prob_seed=333):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
            debug (bool): if True, only one data id is selected from the dataset
        """
        super().__init__(img_size)
        self.data_dir = data_dir
        self.json_file = json_file
        self.coco = COCO(self.data_dir+'/Argoverse-HD/annotations/'+self.json_file)
        self.ids = self.coco.getImgIds()
        self.seq_dirs = self.coco.dataset['seq_dirs']
        self.class_ids = sorted(self.coco.getCatIds())
        # {0: {'id': 0, 'name': 'person'}, 1: {'id': 1, 'name': 'bicycle'}, 2: {'id': 2, 'name': 'car'},
        # 3: {'id': 3, 'name': 'motorcycle'}, 4: {'id': 4, 'name': 'bus'}, 5: {'id': 5, 'name': 'truck'},
        # 6: {'id': 6, 'name': 'traffic_light'}, 7: {'id': 7, 'name': 'stop_sign'}}
        self._classes = self.coco.cats
        self.name = name
        self.max_labels = 50
        self.img_size = img_size
        self.preproc = preproc
        self.annotations_onex = self._load_coco_annotations()
        self.annotations_twox = self._load_coco_annotations_twox()
        self.annotations = None
        self.imgs = None

        self.speed_prob = speed_prob
        self.speed_to_annos = {
            SpeedType.ONEX.value: self.annotations_onex,
            SpeedType.TWOX.value: self.annotations_twox,
        }
        np.random.seed(speed_prob_seed)
        assert isinstance(speed_prob, dict), "speed_prob must be a dictionary."
        assert sum([v for v in speed_prob.values()]) == 1, "The value sum of speed_prob must be 1."

    def __len__(self):
        return len(self.ids)

    def __del__(self):
        if self.imgs:
            del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def _load_coco_annotations_twox(self):
        return [self.load_anno_from_ids_twox(_ids) for _ids in self.ids]

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        im_name = im_ann['name']
        im_sid = im_ann['sid']

        seq_len = len(self.ids)


        #################future  annotation#############

        ## front fid
        if self.coco.dataset['images'][int(id_)]['fid'] == 0:
            im_ann_support = im_ann

        ## back seq fid
        elif int(id_) == seq_len-1:
            im_ann_support = im_ann

        
        ## back fid
        elif self.coco.dataset['images'][int(id_ + 1)]['fid'] == 0:
            im_ann_support = im_ann

        else:
            im_ann_support = self.coco.loadImgs(id_ - 1)[0]

        im_name_support = im_ann_support['name']
        im_sid_support = im_ann_support['sid']



        ## back seq fid
        if id_ in [seq_len-1, seq_len-2]:
            anno_ids = self.coco.getAnnIds(imgIds=[int(seq_len)], iscrowd=False)
        ## back fid
        else:
            if self.coco.dataset['images'][int(id_)]['fid'] == 0:
                anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)

        ## front fid
            elif self.coco.dataset['images'][int(id_ + 1)]['fid'] == 0:
                anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)

            else:
                anno_ids = self.coco.getAnnIds(imgIds=[int(id_ + 1)], iscrowd=False)

        annotations = self.coco.loadAnns(anno_ids)

        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width-1, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height-1, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)


        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid], im_name)
        support_file_name = os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid_support], im_name_support)



        #################support  annotation#############
        support_anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        support_annotations = self.coco.loadAnns(support_anno_ids)

        support_objs = []
        for obj1 in support_annotations:
            x1 = np.max((0, obj1["bbox"][0]))
            y1 = np.max((0, obj1["bbox"][1]))
            x2 = np.min((width-1, x1 + np.max((0, obj1["bbox"][2]))))
            y2 = np.min((height-1, y1 + np.max((0, obj1["bbox"][3]))))
            if obj1["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj1["clean_bbox"] = [x1, y1, x2, y2]
                support_objs.append(obj1)

        support_num_objs = len(support_objs)


        support_res = np.zeros((support_num_objs, 5))

        for ix, obj1 in enumerate(support_objs):
            support_cls = self.class_ids.index(obj1["category_id"])
            support_res[ix, 0:4] = obj1["clean_bbox"]
            support_res[ix, 4] = support_cls

        support_r = min(self.img_size[0] / height, self.img_size[1] / width)
        support_res[:, :4] *= support_r

        support_file_name = os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid_support], im_name_support)

        return (res, support_res, img_info, resized_info, file_name, support_file_name)

    def load_anno_from_ids_twox(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        im_name = im_ann['name']
        im_sid = im_ann['sid']

        seq_len = len(self.ids)


        #################future  annotation#############

        # deal with image info of the case using still annotations
        ## front fid
        # the first frame of a video
        if self.coco.dataset['images'][int(id_)]['fid'] == 0:
            im_ann_support = im_ann

        ## back seq fid
        # the last frame of the last video
        elif int(id_) == seq_len-1:
            im_ann_support = im_ann

        # the next to last frame of the last video
        elif int(id_) == seq_len-2:
            im_ann_support = self.coco.loadImgs(id_ - 1)[0]
        
        ## back fid
        # the last frame of videos except the last video
        elif self.coco.dataset['images'][int(id_ + 1)]['fid'] == 0:
            im_ann_support = im_ann

        # the second frame of a video
        elif self.coco.dataset['images'][int(id_)]['fid'] == 1:
            im_ann_support = self.coco.loadImgs(id_ - 1)[0]

        # the next to last frame of videos except the last video
        elif self.coco.dataset['images'][int(id_ + 2)]['fid'] == 0:
            im_ann_support = self.coco.loadImgs(id_ - 1)[0]

        # other cases
        else:
            im_ann_support = self.coco.loadImgs(id_ - 2)[0]

        im_name_support = im_ann_support['name']
        im_sid_support = im_ann_support['sid']


        # deal with annotations info of the case using still annotations
        ## back seq fid
        # the last three frames of the last video
        if id_ in [seq_len-1, seq_len-2, seq_len-3]:
            anno_ids = self.coco.getAnnIds(imgIds=[int(seq_len)], iscrowd=False)
        ## back fid
        else:
            if self.coco.dataset['images'][int(id_)]['fid'] == 0:
                anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)

        ## front fid
            elif self.coco.dataset['images'][int(id_ + 1)]['fid'] == 0:
                anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)

            # the second frame of videos except the last video
            elif self.coco.dataset['images'][int(id_)]['fid'] == 1:
                anno_ids = self.coco.getAnnIds(imgIds=[int(id_ + 1)], iscrowd=False)

            # the second to last frame of videos except the last video
            elif self.coco.dataset['images'][int(id_ + 2)]['fid'] == 0:
                anno_ids = self.coco.getAnnIds(imgIds=[int(id_ + 1)], iscrowd=False)

            else:
                anno_ids = self.coco.getAnnIds(imgIds=[int(id_ + 2)], iscrowd=False)

        annotations = self.coco.loadAnns(anno_ids)

        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width-1, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height-1, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)


        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid], im_name)
        support_file_name = os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid_support], im_name_support)



        #################support  annotation#############
        support_anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        support_annotations = self.coco.loadAnns(support_anno_ids)

        support_objs = []
        for obj1 in support_annotations:
            x1 = np.max((0, obj1["bbox"][0]))
            y1 = np.max((0, obj1["bbox"][1]))
            x2 = np.min((width-1, x1 + np.max((0, obj1["bbox"][2]))))
            y2 = np.min((height-1, y1 + np.max((0, obj1["bbox"][3]))))
            if obj1["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj1["clean_bbox"] = [x1, y1, x2, y2]
                support_objs.append(obj1)

        support_num_objs = len(support_objs)


        support_res = np.zeros((support_num_objs, 5))

        for ix, obj1 in enumerate(support_objs):
            support_cls = self.class_ids.index(obj1["category_id"])
            support_res[ix, 0:4] = obj1["clean_bbox"]
            support_res[ix, 4] = support_cls

        support_r = min(self.img_size[0] / height, self.img_size[1] / width)
        support_res[:, :4] *= support_r

        support_file_name = os.path.join(self.data_dir, 'Argoverse-1.1', 'tracking', self.seq_dirs[im_sid_support], im_name_support)

        return (res, support_res, img_info, resized_info, file_name, support_file_name)


    def load_resized_img(self, index, annotations):
        img = self.load_image(index, annotations)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img


    def load_image(self, index, annotations):
        file_name = annotations[index][4]

        img_file = file_name

        img = cv2.imread(img_file)
        assert img is not None

        return img


    def load_support_resized_img(self, index, annotations):
        img = self.load_support_image(index, annotations)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_support_image(self, index, annotations):
        file_name = annotations[index][5]

        img_file = file_name

        img = cv2.imread(img_file)
        assert img is not None

        return img

    def pull_item(self, index):
        id_ = self.ids[index]

        annotations = self.random_generate_annos()

        res, support_res, img_info, resized_info, _, _ = annotations[index]

        img = self.load_resized_img(index, annotations)
        support_img = self.load_support_resized_img(index, annotations)

        return img, support_img, res.copy(), support_res.copy(), img_info, np.array([id_])

    def random_generate_annos(self):
        random_num = np.random.rand()
        accumulate_num = 0.
        for k, v in self.speed_prob.items():
            accumulate_num += v
            if random_num < accumulate_num:
                speed_type_name = k
                break
        annotations = copy.deepcopy(self.speed_to_annos[speed_type_name])
        return annotations

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up \
        and pre-processed.
        Args:
            index (int): data index
        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data. \
                The shape is :math:`[self.max_labels, 5]`. \
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w, nh, nw, dx, dy.
                h, w (int): original shape of the image
                nh, nw (int): shape of the resized image without padding
                dx, dy (int): pad size
            id_ (int): same as the input index. Used for evaluation.
        """
        img, support_img, target, support_target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:

            img, support_img, target, support_target = self.preproc((img, support_img), (target, support_target), self.input_dim)

        return np.concatenate((img, support_img), axis=0), (target, support_target), img_info, img_id
