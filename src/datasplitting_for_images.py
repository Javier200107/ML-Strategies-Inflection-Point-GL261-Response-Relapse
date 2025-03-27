from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
from itertools import combinations
import tensorflow as tf
import os

N_SPLITS = 4
TEST_SIZE = 0.6
SEED = 42
SAMPLE_SIZE = 10
CURED_GROUPS = [1276, 1285, 1281, 1284, 1382]

class DataSplittingForImages:
    """Class to split data into training and testing sets."""

    @staticmethod
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

    @staticmethod
    def expanding_window_split(ds: tf.data.TFRecordDataset, n_splits: int = 5) -> dict:
        """
        Perform Expanding Window Split on a TFRecordDataset.

        Args:
            ds (tf.data.TFRecordDataset): Dataset to split.
            n_splits (int): Number of splits for expanding windows.

        Returns:
            dict: A dictionary where each key is the split index and each value is a tuple
                containing (training_dataset, validation_dataset).
        """
        data_list = list(ds.as_numpy_iterator())
        for example in data_list[:5]:  # Inspect first 5 examples
            print("Day of study:", example[3])
            print("Type:", type(example[3]))
            if isinstance(example[3], np.ndarray):
                print("Shape:", example[3].shape)


        # Sort by 'days_of_study' (index 3 in tuple from your dataset)
        # Ensure the scalar value is used for sorting
        data_list.sort(key=lambda x: x[3].item() if isinstance(x[3], np.ndarray) else x[3])

        # Calculate split sizes
        total_length = len(data_list)
        split_sizes = [total_length // n_splits] * n_splits
        for i in range(total_length % n_splits):
            split_sizes[i] += 1

        # Generate expanding window splits
        splits = {}
        for i in range(1, n_splits + 1):
            split_index = sum(split_sizes[:i])
            train_data = data_list[:split_index]
            val_data = data_list[split_index:split_index + split_sizes[i - 1]]

            # Convert lists back to TensorFlow datasets
            train_ds = tf.data.Dataset.from_tensor_slices(train_data)
            val_ds = tf.data.Dataset.from_tensor_slices(val_data)

            splits[i] = (train_ds, val_ds)

        return splits


    
