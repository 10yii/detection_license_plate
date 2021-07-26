
# import some common libraries
import numpy as np
import random
import os
# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer


# Register Train Set
def main():
    setup_logger()

    # if coco.json not exist, construct 
    register_coco_instances("license_plate_train", {},
                        "/dataset/license_plates/annotation/train_annotation.coco.json",
                        "/dataset/license_plates/images/")

    register_coco_instances("license_plate_valid", {},
                        "/dataset/license_plates/annotation/valid_annotation.coco.json",
                        "/dataset/license_plates/images/")
    
    #t
    train()



def train():

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("license_plate_train",)
    cfg.DATASETS.TEST = ("license_plate_valid",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 10
    cfg.SOLVER.MAX_ITER = 15 #adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = (10, 15)
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.TEST.EVAL_PERIOD = 500
    
    os.makedirs("result", exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resule=False)
    trainer.train()


if __name__ == '__main__':
    main()