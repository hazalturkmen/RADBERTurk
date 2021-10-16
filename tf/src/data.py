import pandas as pd
import numpy as np
from bert_tokenizer import load_list
import tensorflow as tf


class RAData(tf.keras.utils.Sequence):
    def __init__(self, csv_path, list_path, batch_size=32, num_classes=None, shuffle=True):
        self.batch_size = batch_size

        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.df = pd.read_excel(csv_path)
        self.indices = self.df.index.tolist()
        self.df = self.df['sonuc']
        self.df = self.df.str.replace('VAR', '1')
        self.df = self.df.str.replace('yok', '0')
        self.df = self.df.str.replace('YOK', '0')
        self.df = self.df.str.replace('SERI DISI', '2')
        self.labels = self.df.tolist()
        self.labels = list(map(int, self.labels))
        self.encoded_imp = load_list(path=list_path)

    def __len__(self):
        return (np.ceil(len(self.encoded_imp) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.encoded_imp[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        text = []
        label = []

        for i in range(0, len(batch_x)):
            text.append(batch_x[i])
            label.append(batch_y[i])
        maxlen = max([len(x) for x in text])
        attention_mask = np.asarray([[1] * len(sent) + [0] * (maxlen - len(sent)) for sent in text])
        x = np.asarray(
            tf.keras.preprocessing.sequence.pad_sequences(list(text), maxlen=maxlen, padding='post', value=0.0),
            dtype=np.int64)
        encoded_label = self.__get_output(label, 3)

        return {'input_ids': tf.convert_to_tensor(x),
                'attention_mask': tf.convert_to_tensor(attention_mask)}, tf.convert_to_tensor(encoded_label)
