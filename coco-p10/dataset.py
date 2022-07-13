import contextlib
import copy
import io
import json
import os
from copy import deepcopy

import numpy as np
import torch
from cvpods.data.build import (SAMPLERS, Infinite, comm, logger,
                               trivial_batch_collator, worker_init_reset_seed)
from cvpods.data.datasets import COCODataset
# from cvpods.data.datasets.paths_route import _PREDEFINED_SPECIAL_COCO
from cvpods.data.detection_utils import (annotations_to_instances,
                                         check_image_size, read_image)


class UnlabeledCOCO:
    def __init__(self, 
                 root='datasets/coco/unlabeled2017', 
                 anno='datasets/coco/annotations/image_info_unlabeled2017.json'):
        with contextlib.redirect_stdout(io.StringIO()):
            self.dataset_dicts = self.load_image_infos(anno,root)
        
    def load_image_infos(self, json_file, root):
        from pycocotools.coco import COCO

        coco_api = COCO(json_file)

        # sort indices for reproducible results
        img_ids = sorted(coco_api.imgs.keys())
        imgs = coco_api.loadImgs(img_ids)

        dataset_dicts = []

        for img_dict in imgs:
            record = {}
            record["file_name"] = os.path.join(root,
                                               img_dict["file_name"])
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            record["image_id"] = img_dict["id"]

            record["annotations"] = None
            dataset_dicts.append(record)
        return dataset_dicts

class PartialCOCO:
    COCO_FULL=None
    COCO_RANDOM_IDX=None
    def __init__(self,percentage=10.0,seed=1,supervised=True,sup_file='DenseTeacher/COCO_Division/COCO_supervision.txt'):
        split_percentage=float(percentage)
        seed=int(seed)
        if PartialCOCO.COCO_FULL is None:
            from cvpods.configs import BaseDetectionConfig as DetCfg
            PartialCOCO.COCO_FULL=COCODataset(DetCfg(),'coco_2017_train').dataset_dicts
        if PartialCOCO.COCO_RANDOM_IDX is None:
            with open(sup_file,'r') as COCO_sup_file:
                PartialCOCO.COCO_RANDOM_IDX = json.load(COCO_sup_file)
        
        self.whole_size = len(PartialCOCO.COCO_FULL)
        labeled_idx = PartialCOCO.COCO_RANDOM_IDX[str(split_percentage)][str(seed)]
        assert len(labeled_idx) == int(self.whole_size*split_percentage/100), "Number of READ_DATA is mismatched."
        if supervised:
            self.dataset_dicts=[PartialCOCO.COCO_FULL[x] for x in labeled_idx]
            logger.info(f'Use {len(self.dataset_dicts)/self.whole_size*100.0:.2f}% data as LABELED.')
            logger.info(f'Use seed={seed}.')
        else:
            indexes = set(range(self.whole_size))
            unlabeled_idx=indexes-set(labeled_idx)
            self.dataset_dicts=[PartialCOCO.COCO_FULL[x] for x in unlabeled_idx]
            logger.info(f'Use {len(self.dataset_dicts)/self.whole_size*100.0:.2f}% data as UNLABELED.')
            logger.info(f'Use seed={seed}.')

class FullCOCO(COCODataset):
    def __init__(self):
        from cvpods.configs import BaseDetectionConfig as DetCfg
        super().__init__(DetCfg(),'coco_2017_train')
       
class SemiTrain:
    def __init__(self,sup_set,unsup_set,sup_aug,unsup_aug):
        self.data_format='BGR'
        self.weakAug = sup_aug
        self.strongAug = unsup_aug
        
        if isinstance(sup_set,list):
            self.sup_dicts=[]
            for _supdata in sup_set:
                self.sup_dicts.extend(_supdata.dataset_dicts)
        else:
            self.sup_dicts=sup_set.dataset_dicts
        
        if isinstance(unsup_set,list):
            self.unsup_dicts=[]
            for _unsupdata in unsup_set:
                self.unsup_dicts.extend(_unsupdata.dataset_dicts)
        else:
            self.unsup_dicts=unsup_set.dataset_dicts
        
        self.size=len(self.sup_dicts)
        self.size_sup=len(self.sup_dicts)
        self.size_unsup=len(self.unsup_dicts)
        logger.info(f'Successfully loaded {self.size_sup} sup images, {self.size_unsup} unsup images.')
        self._set_group_flag_after_init()

    def __getitem__(self, index):
        index_sup = index
        index_unsup = np.random.choice(self.size_unsup)

        dataset_dict_unsup = copy.deepcopy(self.unsup_dicts[index_unsup])
        dataset_dict_sup = copy.deepcopy(self.sup_dicts[index_sup])
        image_unsup = read_image(dataset_dict_unsup["file_name"], format=self.data_format)
        image_sup = read_image(dataset_dict_sup["file_name"], format=self.data_format)
        check_image_size(dataset_dict_unsup, image_unsup)
        check_image_size(dataset_dict_sup, image_sup)

        if "annotations" in dataset_dict_sup:
            annotations_sup = dataset_dict_sup.pop("annotations")
            annotations_sup = [
                ann for ann in annotations_sup if ann.get("iscrowd", 0) == 0]
        else:
            annotations_sup = None
            
        annotations_unsup = None

        # apply transfrom
        image_unsup_weak, annotations_unsup = self.weakAug(
            image_unsup, annotations_unsup)
        image_unsup_strong, annotations_unsup = self.strongAug(
            image_unsup_weak, annotations_unsup)

        image_sup_weak, annotations_sup = self.weakAug(
            image_sup, annotations_sup)
        image_sup_strong, annotations_sup = self.strongAug(
            image_sup_weak, annotations_sup)

        image_shape_sup = image_sup_weak.shape[:2]
        instances_sup = annotations_to_instances(
            annotations_sup, image_shape_sup
        )
        dataset_dict_sup['instances'] = instances_sup

        data_unsup_weak = deepcopy(dataset_dict_unsup)
        data_unsup_strong = deepcopy(dataset_dict_unsup)
        data_sup_weak = deepcopy(dataset_dict_sup)
        data_sup_strong = deepcopy(dataset_dict_sup)

        data_unsup_weak["image"] = torch.as_tensor(
            np.ascontiguousarray(image_unsup_weak.transpose(2, 0, 1)))
        data_unsup_strong["image"] = torch.as_tensor(
            np.ascontiguousarray(image_unsup_strong.transpose(2, 0, 1)))
        data_sup_weak["image"] = torch.as_tensor(
            np.ascontiguousarray(image_sup_weak.transpose(2, 0, 1)))
        data_sup_strong["image"] = torch.as_tensor(
            np.ascontiguousarray(image_sup_strong.transpose(2, 0, 1)))
        del image_sup_weak
        del image_sup_strong
        del image_unsup_weak
        del image_unsup_strong

        return data_unsup_weak, data_unsup_strong, data_sup_weak, data_sup_strong

    def __len__(self):
        return self.size
    
    def _set_group_flag_after_init(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.aspect_ratios = np.zeros(self.size, dtype=np.uint8)
        if "width" in self.sup_dicts[0] and "height" in self.sup_dicts[0]:
            for i in range(self.size_sup):
                dataset_dict = self.sup_dicts[i]
                if dataset_dict['width'] / dataset_dict['height'] > 1:
                    self.aspect_ratios[i] = 1


def build_train_loader(cfg):

    # For simulate large batch training
    num_devices = comm.get_world_size()
    rank = comm.get_rank()

    # use subdivision batchsize
    images_per_minibatch = cfg.SOLVER.IMS_PER_DEVICE // cfg.SOLVER.BATCH_SUBDIVISIONS

    sup_sets=[]
    for dataset_item in cfg.DATASETS.SUPERVISED:
        if len(dataset_item)>1 and isinstance(dataset_item[1],dict):
            sup_sets.append(dataset_item[0](**(dataset_item[1])))
        else:
            sup_sets.append(dataset_item[0]())
            
    unsup_sets=[]
    for dataset_item in cfg.DATASETS.UNSUPERVISED:
        if len(dataset_item)>1 and isinstance(dataset_item[1],dict):
            unsup_sets.append(dataset_item[0](**(dataset_item[1])))
        else:
            unsup_sets.append(dataset_item[0]())
    
    AUGs=cfg.INPUT.AUG.TRAIN_PIPELINES
    AUG_SUP=AUGs.SUPERVISED
    sup_augmentation=AUG_SUP[0](**AUG_SUP[1]) if len(AUG_SUP)>1 else AUG_SUP[0]()
    AUG_UNSUP=AUGs.UNSUPERVISED
    unsup_augmentation=AUG_UNSUP[0](**AUG_UNSUP[1]) if len(AUG_UNSUP)>1 else AUG_UNSUP[0]()
        
    dataset=SemiTrain(sup_sets,unsup_sets,sup_augmentation,unsup_augmentation)
    
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger.info("Using training sampler {}".format(sampler_name))
    

    assert sampler_name in SAMPLERS, "{} not found in SAMPLERS".format(sampler_name)
    if sampler_name == "TrainingSampler":
        sampler = SAMPLERS.get(sampler_name)(len(dataset))
    elif sampler_name == "InferenceSampler":
        sampler = SAMPLERS.get(sampler_name)(len(dataset))
    elif sampler_name == "DistributedGroupSampler":
        sampler = SAMPLERS.get(sampler_name)(
            dataset, images_per_minibatch, num_devices, rank)

    if cfg.DATALOADER.ENABLE_INF_SAMPLER:
        sampler = Infinite(sampler)
        logger.info("Wrap sampler with infinite warpper...")

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_minibatch,
        sampler=sampler,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )

    return data_loader
