import tensorflow as tf
from tensorflow import keras


class Distiller(keras.Model):
    def __init__(self, teacher, student, class_weight_dict=None):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.class_weight_dict = class_weight_dict
        self.augmenter = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomZoom(0.05),
        ])

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha=0.5, temperature=3.0, use_augmentation=True):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
        self.use_augmentation = use_augmentation

    def train_step(self, data):
        x, y = data

        # 3. Augmentation anwenden (Nur im Training!)
        if self.use_augmentation:
            x = self.augmenter(x, training=True)

        # 1. Forward Pass Teacher
        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # 2. Forward Pass Student
            student_predictions = self.student(x, training=True)

            # 3. Student Loss mit Gewichtung
            # student_loss_fn MUSS mit reduction=Reduction.NONE kompiliert sein!
            student_loss = self.student_loss_fn(y, student_predictions)

            if self.class_weight_dict is not None:
                # Gewichte aus Dictionary holen
                weights = tf.constant([self.class_weight_dict[i] for i in range(len(self.class_weight_dict))],
                                      dtype=tf.float32)
                sample_weights = tf.gather(weights, y)
                student_loss = student_loss * sample_weights

            student_loss = tf.reduce_mean(student_loss)

            # 4. Distillation Loss
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )

            loss = ((1-self.alpha) * distillation_loss) + ( self.alpha * student_loss)

        # Gradienten & Update
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Metriken aktualisieren
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y, student_predictions)

        return {m.name: m.result() for m in self.metrics if m.name != "loss"} | {
            "loss": loss,
            "student_loss": student_loss,
            "distillation_loss": distillation_loss
        }

    def test_step(self, data):
        x, y = data
        y_prediction = self.student(x, training=False)
        student_loss = self.student_loss_fn(y, y_prediction)

        # NEU: Keras 3 kompatibles Metrik-Update
        for metric in self.metrics:
            if metric.name != "loss":
                metric.update_state(y, y_prediction)

        results = {m.name: m.result() for m in self.metrics if m.name != "loss"}
        results.update({
            "student_loss": student_loss,
            "loss": student_loss  # WICHTIG: Setzt den offiziellen Keras-Loss auf den Student-Loss
        })
        return results

    def call(self, inputs):
        """Erlaubt es, model.predict() direkt aufzurufen."""
        return self.student(inputs)

    def get_config(self):
        config = super().get_config()
        # Wir schließen die Modelle aus der Konfiguration aus, da sie
        # als Objekte nicht direkt serialisiert werden können.
        return config