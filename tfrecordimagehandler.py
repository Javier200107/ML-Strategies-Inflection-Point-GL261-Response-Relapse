import tensorflow as tf
import numpy as np

class TFRecordImageHandler:
    def __init__(self, tfrecord_file, batch_size=32, shuffle=True, augment=False):
        self.tfrecord_file = tfrecord_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.dataset = self._load_dataset()
        self.length = self.__len__()

    def _parse_function(self, proto):
        # Definir la descripción de las características
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

        
        # Decodificar la imagen
        image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
        image = tf.reshape(image, [parsed_features['height'], parsed_features['width'], 3])
        image = tf.image.rgb_to_grayscale(image)
        # Redimensionar la imagen a un tamaño uniforme (por ejemplo, 256x256)
        image = tf.image.resize(image, [256, 256])

        # Decodificar la máscara
        mask = tf.io.decode_raw(parsed_features['mask'], tf.uint8)
        mask = tf.reshape(mask, [parsed_features['mask_height'], parsed_features['mask_width'], 1])

        # Redimensionar la máscara a un tamaño uniforme (por ejemplo, 256x256)
        mask = tf.image.resize(mask, [256, 256])

        self._normalize(image, mask)

        # Combine image and mask into a 2-channel image
        image = tf.concat([image, mask], axis=-1)

        # Convertir el group_name a one-hot encoding
        group_name_one_hot = tf.one_hot(parsed_features['group_name'], depth=3)  # 3 es el número de clases
        
        # return image, mask, parsed_features['group_name'], parsed_features['mouse_id'], parsed_features['day_of_study']
        return image, group_name_one_hot

    # def _normalize(self, image, group_name): # self, image, mask, group_name, mouse_id, day_of_study
    #     # Convertir la imagen a float y normalizarla al rango [0, 1]
    #     image = tf.cast(image, tf.float32) / 255.0
    #     # Convertir la máscara a float y normalizarla al rango [0, 1]
    #     #mask = tf.cast(mask, tf.float32) / 255.0
        
    #     #return image, mask, group_name, mouse_id, day_of_study
    #     return image, group_name

    def _normalize(self, image, mask):
        # Convertir la imagen a float y normalizarla al rango [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        # Convertir la máscara a float y normalizarla al rango [0, 1]
        mask = tf.cast(mask, tf.float32) / 255.0
        return image, mask

    def _augment(self, image, group_name): # self, image, mask, group_name, mouse_id, day_of_study
        # Aumentación: flipping horizontal y brillo aleatorio
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            #mask = tf.image.flip_left_right(mask)
        image = tf.image.random_brightness(image, max_delta=0.1)

        #return image, mask, group_name, mouse_id, day_of_study
        return image, group_name

    def _load_dataset(self):
        # Cargar el archivo TFRecord y aplicar los mapeos
        dataset = tf.data.TFRecordDataset(self.tfrecord_file)
        dataset = dataset.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        #dataset = dataset.map(self._normalize, num_parallel_calls=tf.data.AUTOTUNE)

        if self.augment:
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        # Aplicar batch y prefetch para eficiencia
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def __len__(self):
        size = sum(1 for _ in tf.data.TFRecordDataset(self.tfrecord_file))
        return int(np.ceil(size / self.batch_size))
