import os
import json
import cv2
import Path

def get_category_list(json_dir):
    json_dir_path = Path(json_dir)
    category_list = []
    for file in json_dir_path.rglob('*.json'):
        filename = str(file)
        with open(filename, 'r') as f:
            temp = json.loads(f.read())
            for i in temp:
                category_list.append(i['classification']['code'])

    return list(set(category_list))


def convert2coco():

    # Check whetever coco.json is  exist.
    config = {}
    with open('config.json', 'r') as infile:
        config = json.load(infile.read())

    annotations_path    = config['TRAIN_ANNOTATIONS_PATH']
    images_path         = config['TRAIN_IMAGES_PATH']
    train_annotation_file = config['TRAIN_COCO_JSON_FILE']
    valid_annotation_file = config['VALID_COCO_JSON_FILE'] 
    # 
    # Refresh coco json file
    if os.path.isfile(train_annotation_file):
        os.remove(train_annotation_file)


    dataset = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    # get category list 
    lst_name = get_category_list(annotations_path)
    category_list = []
    for idx, ctg in enumerate(lst_name):
        category_list.append({
            'id': idx,
            'name': ctg,
            'supercategory': "object"
        })
        category_dict['ctg'] = idx

    images        = Path(images_path)
    im_li = []

    for i, f in enumerate(images.rglog('*.jpg')):
        filename = str(f)
        img =  cv2.imread(filename)
        if img is not None:
            h,w,_ = img.shape
        
            img_data = {
                "id": i, 
                "width": w, 
                "height": h, 
                "file_name": filename.split('/')[-1],
                "license": 0, 
            }
            im_li.append(img_data)

    train_imgs = im_li[:int(config['VALID_RATIO']* len(im_li))]
    valid_imgs = im_li[int(config['VALID_RATIO'] * len(im_li)):]


    train_annos = []
    valid_annos = []
    # train set
    for im in train_imgs:
        train_annos.join(extract_annos(im,annotations_path))

    for im in valid_imgs:
        valid_annos.join(extract_annos(im,annotations_path))


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

    with open(train_annotation_file, 'w') as outfile:
        json.dump(train_dataset, outfile)
    
    with open(valid_annotation_file, 'w') as outfile:
        json.dump(valid_dataset, outfile)

    # if not exist, construct json file
    

    # write 



def extract_annos(im, path):
    json_file = path + '/' + im['file_name'].replace('.jpg', '.json')

    with open(json_file, 'r') as f:
        data = json.load(f.read())

        for anno in data:
            anno_data = {
                "id" : anno_id,
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
            anno_id += 1

def main():
    global anno_id
    global category_dict
    category_dict = {}
    anno_id = 0
    convert2coco()

if __name__ == '_main_':
    main()
