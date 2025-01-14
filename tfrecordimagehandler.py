import tensorflow as tf
import numpy as np

class TFRecordImageHandler:
    def __init__(self, tfrecord_file, batch_size=32, shuffle=False, augment=False):
        self.tfrecord_file = tfrecord_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.dataset = self._load_dataset()
        self.length = self.__len__()

    def _parse_function(self, proto):
        # Define the description of the features
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'mask': tf.io.FixedLenFeature([], tf.string),
            'mask_height': tf.io.FixedLenFeature([], tf.int64),
            'mask_width': tf.io.FixedLenFeature([], tf.int64),
            'group_name': tf.io.FixedLenFeature([], tf.int64),
            'mouse_id': tf.io.FixedLenFeature([], tf.int64),
            'day_of_study': tf.io.FixedLenFeature([], tf.int64),
        }
        parsed_features = tf.io.parse_single_example(proto, feature_description)

        # Decode the image
        image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
        image = tf.reshape(image, [parsed_features['height'], parsed_features['width'], 3])
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize(image, [256, 256])

        # Decode the mask
        mask = tf.io.decode_raw(parsed_features['mask'], tf.uint8)
        mask = tf.reshape(mask, [parsed_features['mask_height'], parsed_features['mask_width'], 1])
        mask = tf.image.resize(mask, [256, 256])

        # Combine image and mask into a 2-channel image
        image = tf.concat([image, mask], axis=-1)

        # Cast group_name to float
        group_name = tf.cast(parsed_features['group_name'], tf.float32)

        # Cast mouse_id to float
        mouse_id = tf.cast(parsed_features['mouse_id'], tf.float32)

        # Cast day_of_study to float
        day_of_study = tf.cast(parsed_features['day_of_study'], tf.float32)
        
        return image, group_name, mouse_id, day_of_study

    def _normalize(self, image):
        # Normalize image to [0, 1] range
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def _augment(self, image, group_name, mouse_id, day_of_study):
        # Apply augmentation only to the image
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)

        # Return the augmented image along with unchanged group_name, mouse_id, and day_of_study
        return image, group_name, mouse_id, day_of_study

    def _load_dataset(self):
        # Load the TFRecord file and apply mappings
        dataset = tf.data.TFRecordDataset(self.tfrecord_file)
        dataset = dataset.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)

        # Normalize before augmenting
        dataset = dataset.map(lambda img, grp, mid, day: (self._normalize(img), grp, mid, day), 
                            num_parallel_calls=tf.data.AUTOTUNE)

        if self.augment:
            dataset = dataset.map(lambda img, grp, mid, day: (*self._augment(img, grp), mid, day),
                                num_parallel_calls=tf.data.AUTOTUNE)

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        # Apply batching and prefetching for efficiency
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

    def __len__(self):
        # Contar los ejemplos directamente en el archivo TFRecord
        return sum(1 for _ in tf.data.TFRecordDataset(self.tfrecord_file))

