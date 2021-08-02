
# import some common libraries
import os
import json
import random
import  cv2
from pathlib import Path

# import some common detectron2 utilities
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultTrainer



#import the COCO Evaluator to use the COCO Metrics
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


# Register Train Set
def main():
    setup_logger()


    # if coco.json not exist, construct 
    with open('config.json' ,'r' ) as file:
        config = json.load(file)

        Image_Path = config['TRAIN_IMAGES_PATH']
        Train_Annos_Path = config['TRAIN_COCO_JSON_FILE']
        Test_Annos_Path = config['TEST_COCO_JSON_FILE']
        # if coco.json not exist, construct 
        register_coco_instances("license_plate_train", {},
                            Train_Annos_Path,
                            Image_Path)

        register_coco_instances("license_plate_test", {},
                            Test_Annos_Path,
                            Image_Path)
    
    #t
    evaluation()



def evaluation():

    

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("license_plate_train",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')  #사용할 모델
    cfg.DATASETS.TEST = ("license_plate_test",)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  
    predictor = DefaultPredictor(cfg) 


    test_metadata = MetadataCatalog.get("license_plate_train")
    dataset_dicts = DatasetCatalog.get("license_plate_train") 


    with open('config.json' ,'r' ) as file:
        config = json.load(file)

        Image_Path = config['TRAIN_IMAGES_PATH']
        for imageName in random.sample(dataset_dicts,3):
            path = os.path.join(Image_Path,imageName['file_name'])
            im = cv2.imread(path)
            print(imageName)
            outputs = predictor(im)

            v = Visualizer(im[:, :, ::-1],
                        metadata=test_metadata,
                        scale=1.0
                        )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imwrite('./result/' + path.split('/')[-1], out.get_image()[:, :, ::-1])


    print("Evaluation Clear")
    

if __name__ == '__main__':
    main()