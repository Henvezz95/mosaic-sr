
import json
import cv2
import hashlib
import os
import sys
from tqdm import tqdm
from glob import glob
import numpy as np
import yaml
import getopt
import sys
import tensorflow as tf
from functions import convert_to_defect_dictionary, image_example

sys.path.insert(0,'./')
from utils.utils import Union, get_percentile_mask, max_distance_from_center, safe_make_folder


def parse_inputs(file_path, argv):
    config_path = None
    k_index = None
    file_name = file_path.split('/')[-1]
    try:
        opts, _ = getopt.getopt(argv, "hc:k:", ["cfile=", "kindex="])
    except getopt.GetoptError:
        print(file_name, '-c <configfile> -k <k_fold_index>')
        print('The configuration file must be in yaml format. K indicates which k-fold interation has to be generated. K is only needed if k-fold is activated')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(file_name, '-c <configfile> -k <k_fold_index>')
            print('The configuration file must be in yaml format. K indicates which k-fold interation has to be generated. K is only needed if k-fold is activated')
            sys.exit()
        elif opt in ("-c", "--cfile"):
            config_path = arg
        elif opt in ("-k", "--kindex"):
            k_index = arg

    if config_path == None:
        print(file_name, '-c <configfile> -k <k_fold_index>')
        print('The configuration file must be in yaml format. K indicates which k-fold interation has to be generated. K is only needed if k-fold is activated')
        sys.exit(2)

    return config_path, int(k_index)

if __name__ == "__main__":    
    config_path, fold_index = parse_inputs(sys.argv[0], sys.argv[1:])
    
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)

    tfrecord_dst_path = config['output_path']
    safe_make_folder(tfrecord_dst_path)
    safe_make_folder(tfrecord_dst_path+'Train')
    safe_make_folder(tfrecord_dst_path+'Validation')

    dataset_paths = config['images_paths']
    json_paths = config['json_paths']
    for path in json_paths:
        json_paths.extend(glob(path+'*.json'))

    dictionary = {}
    for json_path in json_paths:
        if json_path.split('.')[-1] != 'json':
            continue
        with open(json_path) as json_file:
            data = json.load(json_file)
        dictionary = convert_to_defect_dictionary(data, dictionary=dictionary)

    # Loading file paths
    file_paths = []
    for i in range(len(dataset_paths)):
        folder_name = dataset_paths[i]
        for img_type in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPEG', '*.BMP', '*.JPG']:
            new_paths = glob(folder_name+'/'+img_type) 
            file_paths = Union(file_paths, new_paths)

    salt = config['salt']
    k_folds = config['k_folds']
    train_paths = []
    val_paths = []
    train_split = 1 - 1/k_folds

    for path in file_paths:
        file_name = os.path.basename(path)
        hash_code = int(hashlib.sha256(((file_name+salt).encode('utf-8'))).hexdigest(), 16) 
        hash_code = int((int(hash_code % 10 ** 6)+(10 ** 6)*(fold_index/k_folds))% 10 ** 6)
        if hash_code < train_split * (10 ** 6):
            train_paths.append(path)
        else:
            val_paths.append(path)

    # Generating Training Set Records
    normalized_padding = config['normalized_padding']
    ppe_min = config['ppe_min']
    contrast_min = config['contrast_min']
    k = 0
    counter = 0
    buffer = []
    for path in tqdm(train_paths):
        record_name = str(k).zfill(4)+'_record.tfrecord'
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H,W = img.shape
        file_name = os.path.basename(path)
        annotation = dictionary[os.path.basename(path)]
        for i in range(len(annotation)):
            polygon = annotation[i]['polygon']
            minn = get_percentile_mask(img, polygon, 5)
            maxx = get_percentile_mask(img, polygon, 90)
            ppe = np.float32(annotation[i]['ppe'])
            if ppe < ppe_min or maxx-minn<contrast_min:
                continue
            temp_img = np.copy(img)
            for j in range(len(annotation)):
                if i != j:
                    smaller_poly = annotation[j]['polygon']
                    center_spoly = np.mean(annotation[j]['polygon'],axis=0)
                    smaller_poly = np.int32((smaller_poly-center_spoly)*0.9+center_spoly)
                    temp_img = cv2.fillPoly(temp_img, [smaller_poly], color=temp_img.mean())
                else:
                    continue
            radius = max_distance_from_center(polygon)
            center = np.sum(polygon, axis=0)/len(polygon)
            x0 = int(center[0]-radius-normalized_padding*ppe)
            x1 = int(center[0]+radius+normalized_padding*ppe)
            y0 = int(center[1]-radius-normalized_padding*ppe)
            y1 = int(center[1]+radius+normalized_padding*ppe)

            pad1_1 = max(0, -x0)
            pad1_2 = max(0, x1-W)
            pad2_1 = max(0, -y0)
            pad2_2 = max(0, y1-H)

            temp_img = np.pad(temp_img, ((pad2_1,pad2_2),(pad1_1, pad1_2)), mode='edge')

            x0 += pad1_1
            x1 += pad1_1
            y0 += pad2_1
            y1 += pad2_1

            crop = temp_img[y0:y1, x0:x1]
            polygon_translated = polygon - np.array([x0, y0]) + np.array([pad1_1, pad2_1])
            if ppe > 6.0:
                h,w = crop.shape
                ppe_new = 6
                H_new = int((h/ppe)*ppe_new)
                W_new = int((w/ppe)*ppe_new)
                crop = cv2.resize(crop, (W_new, H_new), interpolation=cv2.INTER_AREA)
                polygon_translated = (polygon_translated/ppe)*ppe_new
                ppe=ppe_new

            mask = np.ones_like(crop)*1e-4
            mask = cv2.fillPoly(mask, [np.int32(polygon_translated)], 1, lineType = cv2.LINE_AA)
            mask = cv2.GaussianBlur(mask, (3,3), 0)
            HD_lap = cv2.Laplacian(crop, -1, ksize=3)
            if HD_lap[mask > 0.5].std()*ppe < 100:
                continue
            tf_example = image_example(crop.astype('uint8'), polygon_translated.astype('int32'), file_name, [ppe])
            buffer.append(tf_example)
            counter+=1
            if counter >= 1000:
                with  tf.io.TFRecordWriter(tfrecord_dst_path+'Train/'+record_name) as wrt:
                    for c in range(counter):
                        wrt.write(buffer[c].SerializeToString())
                k+=1
                print(counter)
                counter = 0
                buffer = []

    with  tf.io.TFRecordWriter(tfrecord_dst_path+'Train/'+record_name) as wrt:
        for c in range(counter):
            wrt.write(buffer[c].SerializeToString())
    print(counter)

    # Generating Training Set Records
    k = 0
    record_name = str(k).zfill(4)+'_record.tfrecord'

    with  tf.io.TFRecordWriter(tfrecord_dst_path+'Validation/'+record_name) as wrt:
        for path in tqdm(val_paths):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            H,W = img.shape
            file_name = os.path.basename(path)
            annotation = dictionary[os.path.basename(path)]
            for i in range(len(annotation)):
                polygon = annotation[i]['polygon']
                ttype = annotation[i]['type']
                minn = get_percentile_mask(img, polygon, 5)
                maxx = get_percentile_mask(img, polygon, 90)
                ppe = np.float32(annotation[i]['ppe'])
                if ppe < ppe_min or maxx-minn<contrast_min:
                    continue
                temp_img = np.copy(img)
                for j in range(len(annotation)):
                    if i != j:
                        smaller_poly = annotation[j]['polygon']
                        center_spoly = np.mean(annotation[j]['polygon'],axis=0)
                        smaller_poly = np.int32((smaller_poly-center_spoly)*0.9+center_spoly)
                        temp_img = cv2.fillPoly(temp_img, [smaller_poly], color=temp_img.mean())
                    else:
                        continue
                radius = max_distance_from_center(polygon)
                center = np.sum(polygon, axis=0)/len(polygon)
                x0 = int(center[0]-radius-normalized_padding*ppe)
                x1 = int(center[0]+radius+normalized_padding*ppe)
                y0 = int(center[1]-radius-normalized_padding*ppe)
                y1 = int(center[1]+radius+normalized_padding*ppe)

                pad1_1 = max(0, -x0)
                pad1_2 = max(0, x1-W)
                pad2_1 = max(0, -y0)
                pad2_2 = max(0, y1-H)

                temp_img = np.pad(temp_img, ((pad2_1,pad2_2),(pad1_1, pad1_2)), mode='edge')

                x0 += pad1_1
                x1 += pad1_1
                y0 += pad2_1
                y1 += pad2_1

                crop = temp_img[y0:y1, x0:x1]
                polygon_translated = polygon - np.array([x0, y0]) + np.array([pad1_1, pad2_1])
                
                if ppe > 6.0:
                    ppe_new = 6
                    h,w = crop.shape
                    H_new = int((h/ppe)*ppe_new)
                    W_new = int((w/ppe)*ppe_new)
                    crop = cv2.resize(crop, (W_new, H_new), interpolation=cv2.INTER_AREA)
                    polygon_translated = (polygon_translated/ppe)*ppe_new
                    ppe = ppe_new
                
                mask = np.ones_like(crop)*1e-4
                mask = cv2.fillPoly(mask, [np.int32(polygon_translated)], 1, lineType = cv2.LINE_AA)
                mask = cv2.GaussianBlur(mask, (3,3), 0)
                HD_lap = cv2.Laplacian(crop, -1, ksize=3)
                if HD_lap[mask > 0.5].std()*ppe < 100:
                    continue
                tf_example = image_example(crop.astype('uint8'), polygon_translated.astype('int32'), file_name, [ppe])
                wrt.write(tf_example.SerializeToString())