from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName if layerName else self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        gradModel = Model(inputs=self.model.input,
                          outputs=[self.model.get_layer(self.layerName).output, self.model.output])
        
        with tf.GradientTape() as tape:
            image = tf.convert_to_tensor(image, dtype=tf.float32)
            convOutputs, predictions = gradModel(image)
            loss = predictions if len(predictions.shape) == 1 else predictions[:, 0]

        
        grads = tape.gradient(loss, convOutputs)
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        
        convOutputs = convOutputs[0].numpy()
        guidedGrads = guidedGrads[0].numpy()
        
        weights = np.mean(guidedGrads, axis=(0, 1))
        cam = np.sum(weights * convOutputs, axis=-1)
        
        h, w = image.shape[1:3]
        heatmap = cv2.resize(cam, (w, h))
        
        heatmap = np.maximum(heatmap, 0)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + eps)
        heatmap = (heatmap * 255).astype("uint8")
        # heatmap = 255 - heatmap  # Invertir: ahora 255 será lo menos importante y 0 lo más
        
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        heatmap = 255 - heatmap  # Invertir: ahora 255 será lo menos importante y 0 lo más

        heatmap = cv2.applyColorMap(heatmap, colormap)
        if len(image.shape) == 2 or image.shape[-1] == 1:
            image = cv2.cvtColor(image.squeeze().astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        return heatmap, output
