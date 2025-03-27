import tensorflow as tf
import numpy as np

class TFRecordMultimodalHandler:
    def __init__(self, tfrecord_file, batch_size=32, shuffle=False, augment=False):
        self.tfrecord_file = tfrecord_file
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.dataset = self._load_dataset()
        self.length = self.__len__()

    def pad_sequence(self, tensor, max_length):
        """Rellena la secuencia con ceros hasta max_length."""
        current_length = tf.shape(tensor)[0]
        padding_size = max_length - current_length
        padding = tf.zeros(tf.concat([[padding_size], tf.shape(tensor)[1:]], axis=0), dtype=tensor.dtype)
        tf.print("PADDING:", tensor.shape, "->", tf.shape(padding))  # 📌 Debugging
        print(tf.concat([tensor, padding], axis=0))
        return tf.concat([tensor, padding], axis=0)

    def _parse_function(self, proto):
        """Parses a single TFRecord example."""
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

        # Decodificar imagen
        image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
        image = tf.reshape(image, [parsed_features['height'], parsed_features['width'], 3])
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.resize(image, [256, 256])

        # Decodificar máscara
        mask = tf.io.decode_raw(parsed_features['mask'], tf.uint8)
        mask = tf.reshape(mask, [parsed_features['mask_height'], parsed_features['mask_width'], 1])
        mask = tf.image.resize(mask, [256, 256])

        # Combinar imagen con máscara
        image = tf.concat([image, mask, mask], axis=-1)

        # Convertir características
        group_name = tf.cast(parsed_features['group_name'], tf.int32)
        day_of_study = tf.cast(parsed_features['day_of_study'], tf.int32)
        mouse_id = tf.cast(parsed_features['mouse_id'], tf.int64)

        map_timesteps = {
            1360: 15, 1263: 56, 1270: 43, 1359: 9, 1258: 9, 1284: 8, 1299: 2, 1380: 47,
            1382: 8, 1361: 12, 1276: 21, 1285: 17, 1261: 19, 1260: 16, 1264: 43, 1383: 43,
            1297: 2, 1281: 12
        }

        #tf.print("Mouse ID:", mouse_id, "Day:", day_of_study)  # 📌 Debugging

        return image, day_of_study, mouse_id, group_name

    def _normalize(self, image):
        """Normaliza la imagen a [0, 1]."""
        return tf.cast(image, tf.float32) / 255.0

    def _load_dataset(self):
        dataset = tf.data.TFRecordDataset(self.tfrecord_file)
        dataset = dataset.map(self._parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Normalizar imágenes
        dataset = dataset.map(lambda img, day, mouse_id, grp: 
            (self._normalize(img), day, mouse_id, grp), num_parallel_calls=tf.data.AUTOTUNE)

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=1000)

        # 📌 **Agrupar por mouse_id**
        dataset = dataset.group_by_window(
            key_func=lambda img, day, mouse_id, grp: tf.cast(mouse_id, tf.int64),  
            reduce_func=lambda key, ds: ds.batch(1000),  
            window_size=1000
        )

        # 📌 **Ordenar por día de estudio**
        def sort_by_day(images, days, mouse_ids, labels):
            sorted_indices = tf.argsort(days, axis=0)
            sorted_images = tf.gather(images, sorted_indices, axis=0)
            sorted_days = tf.gather(days, sorted_indices, axis=0)
            #tf.print("SORTED DAYS:", sorted_days)  # 📌 Debugging
            return sorted_images, sorted_days, mouse_ids[0], labels[0]

        dataset = dataset.map(sort_by_day, num_parallel_calls=tf.data.AUTOTUNE)

        # 📌 **Asegurar que todas las secuencias dentro de un batch tengan la misma longitud**
        def pad_batch(images, days, mouse_id, labels):
            # max_timesteps = tf.shape(images)[0]  # Se usa el max de cada batch
            # print(f"MAX TIMESTEPS IN BATCH for: {mouse_id}={max_timesteps}: {days}")  # 📌 Debugging
            # padded_images = self.pad_sequence(images, max_timesteps)
            # #padded_days = self.pad_sequence(tf.expand_dims(days, -1), max_timesteps)
            return {"image_input": images, "day_input": days, "mouse_id": mouse_id}, labels
        
        def pad_batch2(images, days, mouse_id, labels):
            max_timesteps = 56  # Queremos que todas las secuencias tengan 56 timesteps
            
            # Obtener la cantidad actual de timesteps
            current_timesteps = tf.shape(images)[0]

            # Calcular la cantidad de padding necesaria
            padding_size = max_timesteps - current_timesteps

            # Si ya tiene la longitud máxima, devolver tal cual
            def no_padding():
                return {"image_input": images, "day_input": days, "mouse_id": mouse_id}, labels

            # Si necesita padding, agregar imágenes de ceros
            def apply_padding():
                zero_padding = tf.zeros([padding_size, 256, 256, 3], dtype=images.dtype)
                padded_images = tf.concat([images, zero_padding], axis=0)

                zero_days = tf.zeros([padding_size], dtype=days.dtype)
                padded_days = tf.concat([days, zero_days], axis=0)

                return {"image_input": padded_images, "day_input": padded_days, "mouse_id": mouse_id}, labels

            return tf.cond(current_timesteps < max_timesteps, apply_padding, no_padding)


        dataset = dataset.map(pad_batch, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

    def __len__(self):
        return sum(1 for _ in tf.data.TFRecordDataset(self.tfrecord_file))



# dataset = dataset.group_by_window(
        #     key_func=lambda x, y: tf.cast(x["mouse_id"], tf.int64),  # ✅ Convertimos a int64
        #     reduce_func=lambda key, ds: ds.batch(1),
        #     window_size=1
        # )