import os
import json


 #

def main():

    # check whetever coco.json is  exist.
    dict = {}
    with open('config.json', 'r') as infile:
        dict = json.load(infile)

    annotation_path = dict['TRAIN_ANNOTATION_PATH']
    image_path = dict['TRAIN_IMAGE_PATH']

    if os.path.isfile('/home/appuser/annotation/annotation.coco.json'):
        return
    



if __name__ == '_main_':
    main()
