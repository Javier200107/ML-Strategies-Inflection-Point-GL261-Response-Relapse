import tensorflow as tf
import numpy as np
import os
import random

# Path to data
SRC = 'dataset/'
print('SRC:', SRC)

# Definir las proporciones de train y validation
train_ratio = 0.7
val_ratio = 0.3

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image_path, mask_path):
    img = tf.io.decode_image(tf.io.read_file(image_path), channels=3)
    img_bytes = img.numpy().tobytes()

    mask = tf.io.decode_image(tf.io.read_file(mask_path), channels=1)
    mask_bytes = mask.numpy().tobytes()

    # Get Study, Mice group and Day of study from the path
    path = os.path.normpath(mask_path)

    # Divide el path en partes
    path_parts = path.split(os.sep)

    # Extraer las partes relevantes
    # 0 for Cured Mice, 1 for Control Mice, 2 for Relapse Mice
    if path_parts[1] == 'Cured mice':
        group_name = 0
    elif path_parts[1] == 'Control':
        group_name = 1
    elif path_parts[1] == 'IMS-TMS-TREATED-RELAPSING':
        group_name = 2
    else:
        print(path_parts)
        raise ValueError('Invalid group')
    mouse_id = path_parts[2]       # 'C1281'
    day_of_study = path_parts[3]   # 'day10'

    # Get only the integers in ids
    mouse_id = mouse_id[1:]
    day_of_study = day_of_study[3:]

    feature = {
        'image': _bytes_feature(img_bytes),
        'mask': _bytes_feature(mask_bytes),
        'height': _int64_feature(img.shape[0]),
        'width': _int64_feature(img.shape[1]),
        'mask_height': _int64_feature(mask.shape[0]),
        'mask_width': _int64_feature(mask.shape[1]),
        'group_name': _int64_feature(int(group_name)),
        'mouse_id': _int64_feature(int(mouse_id)),
        'day_of_study': _int64_feature(int(day_of_study))
    }
    
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def create_tfrecord(file_paths, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for img_path, mask_path in file_paths:
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f'Image or mask not found: {img_path}, {mask_path}')
                continue
            writer.write(serialize_example(img_path, mask_path))

def split_dataset(root_dir, train_ratio):
    all_data = []
    
    # Recorrer todas las imágenes y máscaras
    for data_group in os.listdir(root_dir):
        data_group_path = os.path.join(root_dir, data_group)
        if not os.path.isdir(data_group_path):
            continue
        for mice_group in os.listdir(data_group_path):
            mice_group_path = os.path.join(data_group_path, mice_group)
            if not os.path.isdir(mice_group_path):
                continue  # Saltar si no es un directorio
            for dayofstudy in os.listdir(mice_group_path):
                dayofstudy_path = os.path.join(mice_group_path, dayofstudy)
                if not os.path.isdir(dayofstudy_path):
                    continue  # Saltar si no es un directorio
                mri_imgs = os.path.join(dayofstudy_path, 'MRI images')
                if not os.path.exists(mri_imgs):
                    print(f"No se encontró la carpeta {mri_imgs}")
                    continue
                # Número de imágenes
                n_imgs = len(os.listdir(mri_imgs))
                for i in range(n_imgs):
                    img_path = os.path.join(mri_imgs, f'image_s{i + 1}.jpg')
                    mask_path = os.path.join(dayofstudy_path, f'Mask s{i + 1}.jpg')
                    all_data.append((img_path, mask_path))

    # Barajar los datos para que no haya un orden predecible
    random.shuffle(all_data)

    # Dividir en conjuntos de entrenamiento y validación
    train_size = int(len(all_data) * train_ratio)
    train_data = all_data[:train_size]
    val_data = all_data[train_size:]

    return train_data, val_data

if __name__ == '__main__':
    # Dividir el dataset en 70% train y 30% validation
    train_data, val_data = split_dataset(SRC, train_ratio)

    # Crear TFRecords para entrenamiento y validación
    create_tfrecord(train_data, 'train.tfrecords')
    create_tfrecord(val_data, 'val.tfrecords')

    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
