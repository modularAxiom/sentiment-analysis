
from transformers import TFBertForSequenceClassification
import tensorflow as tf

def build_model():
    """
    Builds and compiles the TF-Bert model for sequence classification.

    :return: Compiled TensorFlow model.
    """
    # Load pre-trained BERT model with a classification head
    model = TFBertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,   # Binary Classification
    )

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    return model
