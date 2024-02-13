import json
import numpy as np
import os.path as osp

def split(images,threshold):
    mode2id=dict(train=[], test=[])
    id2img = dict()
    id2mode = dict()
    for i,img in enumerate(images):
        file_name = img['file_name']
        file_name += ".jpg" if file_name[-4:]!='.jpg' else ''
        img['file_name'] = file_name
        filename_splitted = file_name.split("_")
        num = int(filename_splitted[-1].split(".")[0])
        #cat = filename_splitted[0] if len(filename_splitted)==2 else "_".join(filename_splitted[:-1])
        id = img['id']
        id2img[id] = img
        mode = "train" if num<=threshold else "test"
        mode2id[mode].append(id)
        id2mode[id] = mode
    return mode2id, id2mode, id2img
        
if __name__=="__main__":
    ##parameters##
    dataset_name = "NEU_DET"
    test_size = 1/5
    train_ouput_name = "instances_train"
    test_output_name = "instances_test"
    
    output_name = dict(train=train_ouput_name, test=test_output_name)

    cur_dirname = osp.dirname(__file__)
    ann_dir = osp.join(cur_dirname, f"../{dataset_name}/annotations")
    ann_pth = osp.join(ann_dir, "all_samples.json")
    file = json.load(open(ann_pth))
    anns = file['annotations']
    images = file['images']
    cats = file['categories']
    threshold = int(len(images)/len(cats)*(1-test_size))
    mode2id, id2mode, id2img = split(images, threshold)
    train_test = ('train', 'test')
    output_files = dict()
    for mode in train_test:
        ids = mode2id[mode]
        shuffled_ids = np.random.permutation(ids)
        output_file = dict(images=[], annotations=[], categories=cats)
        for id in shuffled_ids:
            img = id2img[id]
            output_file["images"].append(img)
        output_files[mode] = output_file

    for ann in anns:
        id = ann['image_id']
        mode = id2mode[id]
        output_files[mode]['annotations'].append(ann)
    
    for mode in train_test:
        name = output_name[mode]
        save_pth = osp.join(ann_dir, name+".json")
        output_file = output_files[mode]
        with open(save_pth, 'w') as f:
            json.dump(output_file, f,indent=4, separators=(',', ': '))
    
            
        
    
    
    
        
    