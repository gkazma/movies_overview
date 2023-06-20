import pandas as pd
from ast import literal_eval
import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# There are multiple genres per movie
train_df = pd.read_csv("dataset/train_data.csv",usecols=['genres', 'overview'], converters={"genres":literal_eval})
test_df = pd.read_csv("dataset/test_data.csv",usecols=['genres', 'overview'], converters={"genres":literal_eval})
train_df.head()

# Initial train and test split.
test_split = 0.1

train_df, val_df = train_test_split(
    train_df,
    test_size=test_split,
    stratify=train_df["genres"].values,
)

import tensorflow as tf


genres = tf.ragged.constant(train_df["genres"].values)
lookup = tf.keras.layers.StringLookup(output_mode="multi_hot", num_oov_indices=0)
lookup.adapt(genres)
vocab = lookup.get_vocabulary()

print("Vocabulary:\n")
print(vocab)

batch_size = 128

def make_dataset(dataframe, is_train=True):
    labels = tf.ragged.constant(dataframe["genres"].values)
    label_binarized = lookup(labels).numpy()
    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe["overview"].values, label_binarized)
    )
    dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
    return dataset.batch(batch_size)


train_dataset = make_dataset(train_df, is_train=True)
validation_dataset = make_dataset(val_df, is_train=False)
test_dataset = make_dataset(test_df, is_train=False)

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

from tfhub_maps import *

bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'

tfhub_handle_encoder = map_name_to_handle[bert_model_name]
tfhub_handle_preprocess = map_model_to_preprocess[bert_model_name]

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

bert_model = hub.KerasLayer(tfhub_handle_encoder)

def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(lookup.vocabulary_size(), activation="sigmoid")(net)
  return tf.keras.Model(text_input, net)

epochs = 1
steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

init_lr = 3e-5

model = build_classifier_model()

optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

model.compile(
    loss="binary_crossentropy", 
    optimizer=optimizer,
    metrics=[tf.keras.metrics.BinaryAccuracy(), 
             tf.keras.metrics.CategoricalAccuracy(), 
             tf.keras.metrics.Accuracy(), 
             tf.keras.metrics.AUC(), 
             tf.keras.metrics.F1Score(average='macro'), 
             tf.keras.metrics.Precision(), 
             tf.keras.metrics.Recall()]
)

early_stopping_monitor = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=0,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)

history = model.fit(
    train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=[early_stopping_monitor], verbose=1
)

model.evaluate(test_dataset)

import os

model_dir = "models/"
model_name = "model"
model_version = "1"
model_export_path = f"{model_dir}/{model_name}/{model_version}"

invert_stringlookup_layer = tf.keras.layers.StringLookup(vocabulary=vocab, invert=True)

model_for_inference = keras.Sequential([model, 
                                        layers.Lambda(lambda x: tf.round(x)),
                                        layers.Lambda(lambda x: tf.map_fn(lambda y: tf.where(y == 1.0)[..., 0] + 1, x, dtype=(tf.int64))),
                                        invert_stringlookup_layer
                                        ])

tf.saved_model.save(
    model_for_inference,
    export_dir=model_export_path,
)

print(f"SavedModel files: {os.listdir(model_export_path)}")