
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# -------------------------------
# DWTFreqBranch
# -------------------------------
class DWTFreqBranch(layers.Layer):
    def __init__(self, pool_size=(2,2), **kwargs):
        super().__init__(**kwargs)
        self.pool = layers.AveragePooling2D(pool_size=pool_size, strides=pool_size, padding='same')
        self.upsample = None

    def build(self, input_shape):
        h, w = input_shape[1], input_shape[2]
        self.upsample = layers.Resizing(h, w, interpolation='bilinear')
        super().build(input_shape)

    def call(self, x):
        low = self.pool(x)
        low_up = self.upsample(low)
        high = tf.math.abs(x - low_up)
        return layers.Concatenate(axis=-1)([low_up, high])

# -------------------------------
# DMSFA block
# -------------------------------
def dmsfa_block_with_lesion(x, name="DMSFAplus"):
    b1 = layers.Conv2D(128, 1, padding='same', activation='relu', name=f'{name}_b1')(x)
    b2 = layers.Conv2D(128, 3, dilation_rate=2, padding='same', activation='relu', name=f'{name}_b2')(x)
    b3 = layers.Conv2D(128, 5, dilation_rate=3, padding='same', activation='relu', name=f'{name}_b3')(x)

    freq = DWTFreqBranch(name=f'{name}_dwt')(x)
    freq = layers.Conv2D(128, 3, padding='same', activation='relu', name=f'{name}_freq_conv')(freq)

    concat = layers.Concatenate(name=f'{name}_concat')([b1, b2, b3, freq])
    fusion = layers.Conv2D(512, 1, padding='same', activation='relu', name=f'{name}_fusion')(concat)
    fusion = layers.BatchNormalization(name=f'{name}_bn')(fusion)

    gating = layers.GlobalAveragePooling2D()(fusion)
    gating = layers.Dense(128, activation='relu', name=f'{name}_gate_dense1')(gating)
    gating = layers.Dense(4, activation='softmax', name=f'{name}_gate_dense2')(gating)
    gating = layers.Reshape((1,1,4), name=f'{name}_gate_reshape')(gating)

    gated_sum = (gating[...,0:1]*b1 + gating[...,1:2]*b2 +
                 gating[...,2:3]*b3 + gating[...,3:4]*freq)
    gated_proj = layers.Conv2D(512, 1, padding='same', activation='relu', name=f'{name}_gated_proj')(gated_sum)
    fusion = layers.Add(name=f'{name}_fused_sum')([fusion, gated_proj])

    gap = layers.GlobalAveragePooling2D()(fusion)
    gap = layers.Reshape((1,1,512))(gap)
    ch_att = layers.Dense(64, activation='relu')(gap)
    ch_att = layers.Dense(512, activation='sigmoid')(ch_att)
    ch_weighted = layers.Multiply()([fusion, ch_att])

    avg_pool = layers.Lambda(lambda z: tf.reduce_mean(z, axis=-1, keepdims=True))(ch_weighted)
    max_pool = layers.Lambda(lambda z: tf.reduce_max(z, axis=-1, keepdims=True))(ch_weighted)
    sp_concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    sp_att = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(sp_concat)

    lesion_map = layers.Conv2D(64, 3, padding='same', activation='relu')(sp_att)
    lesion_map = layers.Conv2D(1, 3, padding='same', activation='sigmoid', name=f'{name}_lesion')(lesion_map)

    fused_output = layers.Multiply()([ch_weighted, lesion_map])
    fused_output = layers.Add()([fused_output, fusion])
    return fused_output, lesion_map

# -------------------------------
# Build model
# -------------------------------
def build_model_multi_output(input_shape=(224,224,3), num_classes=4):
    inputs = layers.Input(shape=input_shape)
    base = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    x = base.output
    fused_spatial, lesion_map = dmsfa_block_with_lesion(x)
    fused_masked = layers.Multiply()([fused_spatial, lesion_map])
    pooled = layers.GlobalAveragePooling2D()(fused_masked)
    fc = layers.Dense(128, activation='relu')(pooled)
    fc = layers.Dropout(0.5)(fc)
    cls_out = layers.Dense(num_classes, activation='softmax', dtype='float32', name='class_output')(fc)
    model = models.Model(inputs=inputs, outputs=[cls_out, lesion_map])
    return model, base

# -------------------------------
# Custom training model
# -------------------------------
class DMSFAModel(tf.keras.Model):
    def __init__(self, model_core, backbone, cam_loss_weight=0.05, grad_clip=1.0):
        super().__init__()
        self.model_core = model_core
        self.backbone = backbone   #  store EfficientNet
        self.cam_loss_weight = cam_loss_weight
        self.grad_clip = grad_clip
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.metric_acc = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
        self.current_epoch = tf.Variable(0.0, trainable=False)           # track epoch

    def compute_gradcam(self, images, class_idx=None):
        """
        Paper-consistent Grad-CAM (Eq. 34–35)
        """

        #  Use stored backbone
        backbone = self.backbone

        #  Get last conv layer
        last_conv_layer = backbone.get_layer("top_conv")

        #  Build grad model
        grad_model = tf.keras.models.Model(
            inputs=self.model_core.input,
            outputs=[last_conv_layer.output, self.model_core.output[0]]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(images, training=False)

            if class_idx is None:
                class_idx = tf.argmax(predictions, axis=-1)

            one_hot = tf.one_hot(tf.cast(class_idx, tf.int32), predictions.shape[-1])
            selected = tf.reduce_sum(predictions * one_hot, axis=-1)

        # Gradients wrt feature maps
        grads = tape.gradient(selected, conv_outputs)

        weights = tf.reduce_mean(grads, axis=[1,2], keepdims=True)
        cam = tf.reduce_sum(weights * conv_outputs, axis=-1, keepdims=True)

        cam = tf.nn.relu(cam)

        cam = tf.image.resize(cam, tf.shape(images)[1:3])

        minv = tf.reduce_min(cam, axis=[1,2,3], keepdims=True)
        maxv = tf.reduce_max(cam, axis=[1,2,3], keepdims=True)
        cam = (cam - minv) / (maxv - minv + 1e-8)

        return cam

    def train_step(self, data):
        x, (y_true, _) = data
        with tf.GradientTape() as tape:
            cls_pred, lesion_map = self.model_core(x, training=True)
            cls_loss = self.loss_fn(y_true, cls_pred)
            class_idx = tf.argmax(cls_pred, axis=-1)

            # Compute Grad-CAM
            cams = self.compute_gradcam(x, class_idx)

            # Resize lesion_map to match CAM shape
            lesion_map_resized = tf.image.resize(
                lesion_map, tf.shape(cams)[1:3], method='bilinear'
            )

            # ---  Region-focused CAM loss ---
            weight_map = tf.where(lesion_map_resized > 0.5, 1.5, 0.5)
            cam_loss = tf.reduce_mean(weight_map * tf.square(lesion_map_resized - cams))
            cam_loss = tf.clip_by_value(cam_loss, 0.0, 1.0)

            # total_loss = cls_loss + self.cam_loss_weight * cam_loss
            lambda_t = tf.minimum(self.cam_loss_weight, (self.current_epoch / 5.0) * self.cam_loss_weight)
            total_loss = cls_loss + lambda_t * cam_loss

        grads = tape.gradient(total_loss, self.model_core.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.grad_clip)
        self.optimizer.apply_gradients(zip(grads, self.model_core.trainable_variables))
        self.metric_acc.update_state(y_true, cls_pred)
        return {"loss": total_loss, "cls_loss": cls_loss, "cam_loss": cam_loss, "accuracy": self.metric_acc.result()}


    def test_step(self, data):
        x, (y_true, _) = data
        cls_pred, lesion_map = self.model_core(x, training=False)
        cls_loss = self.loss_fn(y_true, cls_pred)
        class_idx = tf.argmax(cls_pred, axis=-1)

        cams = self.compute_gradcam(x, class_idx)

        # Resize lesion_map to match CAM shape
        lesion_map_resized = tf.image.resize(
            lesion_map, tf.shape(cams)[1:3], method='bilinear'
        )

        weight_map = tf.where(lesion_map_resized > 0.5, 1.5, 0.5)
        cam_loss = tf.reduce_mean(weight_map * tf.square(lesion_map_resized - cams))
        cam_loss = tf.clip_by_value(cam_loss, 0.0, 1.0)

        # total_loss = cls_loss + self.cam_loss_weight * cam_loss
        lambda_t = tf.minimum(self.cam_loss_weight, (self.current_epoch / 5.0) * self.cam_loss_weight)
        total_loss = cls_loss + lambda_t * cam_loss

        self.metric_acc.update_state(y_true, cls_pred)
        return {"loss": total_loss, "cls_loss": cls_loss, "cam_loss": cam_loss, "accuracy": self.metric_acc.result()}

    @property
    def metrics(self):
        return [self.metric_acc]

