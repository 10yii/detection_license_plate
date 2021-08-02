import os
import json
import cv2
from tqdm import tqdm
from pathlib import Path
import magic
import re

category_dict = {}

def main():
    global anno_uid # annotation  id
    global category_dict # category dict

    
    print("========= Convert to CoCo json file =========")
    convert2coco()
    print("========= Convert to CoCo json file =========")


def get_category_list(json_dir):
    json_dir_path   = Path(json_dir)
    json_list       = json_dir_path.rglob('*.json')
    length          = len([_ for _ in json_list])
    category_list   = []
    with tqdm(total=length) as pbar:
        for file in json_dir_path.rglob('*.json'):
            filename = str(file)
            with open(filename, 'r') as f:
                temp = json.loads(f.read())
                for i in temp:
                    category_list.append(i['classification']['code'])
                pbar.update(1)

    return list(set(category_list))


def convert2coco():

    # Check whetever coco.json is  exist.

    print("Step1: Open config file")
    config = {}
    with open('config.json', 'r') as infile:
        config = json.load(infile)


    annotations_path    = config['TRAIN_ANNOTATIONS_PATH']
    images_path         = config['TRAIN_IMAGES_PATH']
    
    train_annotation_file = config['TRAIN_COCO_JSON_FILE']
    valid_annotation_file = config['VALID_COCO_JSON_FILE'] 
    test_annotation_file  = config['TEST_COCO_JSON_FILE']
    # 

    # Refresh coco json file
    if os.path.isfile(train_annotation_file):
        os.remove(train_annotation_file)

    if os.path.isfile(valid_annotation_file):
        os.remove(valid_annotation_file)

    if os.path.isfile(test_annotation_file):
        os.remove(test_annotation_file)


    # get category list 
    print("Loading Category List.....")
    lst_name = get_category_list(annotations_path)
    category_list = []
    for idx, ctg in enumerate(lst_name):
        category_list.append({
            'id': idx,
            'name': ctg,
            'supercategory': "object"
        })
        category_dict[ctg] = idx

    print("Categorization Dataset (Train, Valid , Test)...")
    print(f"Image Path: {images_path}")
    images        = Path(images_path).glob('*.png')
    im_li = []
    with tqdm(total=len([_ for _ in images])) as pbar:
        for i, f in enumerate(Path(images_path).glob('*.png'), start=1):
            filename = str(f)
            t = magic.from_file(filename)
            w , h = re.search('(\d+) x (\d+)', t).groups()
            img_data = {
                    "id": i, 
                    "width": int(w), 
                    "height": int(h), 
                    "file_name": filename.split('/')[-1],
                    "license": 0, 
            }
            im_li.append(img_data)
            pbar.update(1)

    print("Total length: %d", len(im_li))


    train_imgs = im_li[:int(config['VALID_RATIO']* len(im_li))]
    valid_imgs = im_li[int(config['VALID_RATIO'] * len(im_li)):int((config['TEST_RATIO'] + config['VALID_RATIO'])* len(im_li))]
    test_imgs  = im_li[int((config['TEST_RATIO'] + config['VALID_RATIO'])* len(im_li)):]

    train_annos = []
    valid_annos = []
    test_annos  = []

    print("Construct CoCo file (Train, Valid , Test)...")

    anno_uid = 0
    # train set
    for im in train_imgs:
        (f, uid) = extract_annos(im,annotations_path, anno_uid)
        anno_uid = uid
        train_annos.extend(f)

    for im in valid_imgs:
        (f, uid) = extract_annos(im,annotations_path, anno_uid)
        anno_uid = uid
        valid_annos.extend(f)    

    for im in test_imgs:
        (f, uid) = extract_annos(im,annotations_path, anno_uid)
        anno_uid = uid
        test_annos.extend(f)

    print("Train data length : %d", len(train_annos))
    print("Valid data length : %d", len(valid_annos))
    print("Test data length : %d", len(test_annos))


    train_dataset = {
        "info": {},
        "licenses": [],
        "categories": category_list,
        "images": train_imgs,
        "annotations": train_annos
    }

    valid_dataset = {
        "info": {},
        "licenses": [],
        "categories": category_list,
        "images": valid_imgs,
        "annotations": valid_annos
    }

    test_dataset = {
        "info": {},
        "licenses": [],
        "categories": category_list,
        "images": test_imgs,
        "annotations": test_annos
    }

    with open(train_annotation_file, 'w') as outfile:
        json.dump(train_dataset, outfile)
    
    print("Train CoCo file is now Made.")
    
    with open(valid_annotation_file, 'w') as outfile:
        json.dump(valid_dataset, outfile)
    print("Valid CoCo file is now Made.")


    with open(test_annotation_file, 'w') as outfile:
        json.dump(test_dataset, outfile)
    print("Test CoCo file is now Made.")



def extract_annos(im, path, uid):
    json_file = path + '/' + im['file_name'].replace('.png', '.json')

    l = []
    with open(json_file, 'r') as f:
        data = json.load(f)

        for anno in data:
            anno_data = {
                "id" : uid,
                "image_id" : im['id'],
                "category_id" : category_dict[anno['classification']['code']],
                "segmentation" : [],
                "bbox" : [anno['label']['data']['x'],
                          anno['label']['data']['y'],
                          anno['label']['data']['width'],
                          anno['label']['data']['height']],
                "area" : anno['label']['data']['width'] * anno['label']['data']['height'] ,
                "iscrowd" : 0
                }   
            uid += 1
            l.append(anno_data)
    return (l, uid)
            



if __name__ == '__main__':
    main()
