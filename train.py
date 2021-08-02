
# import some common libraries
import os
import sys
import json
# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
#import the COCO Evaluator to use the COCO Metrics
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


# Register Train Set
def main():


    
    setup_logger()

    with open('config.json' ,'r' ) as file:
        config = json.load(file)

        Image_Path = config['TRAIN_IMAGES_PATH']
        Train_Annos_Path = config['TRAIN_COCO_JSON_FILE']
        Valid_Annos_Path = config['VALID_COCO_JSON_FILE']
        # if coco.json not exist, construct 
        register_coco_instances("license_plate_train", {},
                            Train_Annos_Path,
                            Image_Path)

        register_coco_instances("license_plate_valid", {},
                            Valid_Annos_Path,
                            Image_Path)
    
    train()



def train():

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'))
    cfg.DATASETS.TRAIN = ("license_plate_train",)
    cfg.DATASETS.TEST = ("license_plate_valid",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml')  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.SOLVER.MAX_ITER = 1500  #adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = (1000, )
    cfg.SOLVER.GAMMA = 0.05
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.TEST.EVAL_PERIOD = 500
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


    evaluator = COCOEvaluator("license_plate_valid", cfg, False, output_dir= cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "license_plate_valid")

    #Use the created predicted model in the previous step
    inference_on_dataset(trainer.model, val_loader, evaluator)
    print("Evaluation Clear")

    

if __name__ == '__main__':
    main()