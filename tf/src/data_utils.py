import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer
from transformers import InputExample, InputFeatures


def read_data(data_path):
    df = pd.read_excel(data_path)
    df["sonuc"] = df["sonuc"].str.replace('VAR', '1')
    df["sonuc"] = df["sonuc"].str.replace('YOK', '0')
    df["sonuc"] = df["sonuc"].str.replace('SERI DISI', '2')
    df['sonuc'] = df['sonuc'].astype(int)
    df['sonuc'] = df['sonuc'].tolist()

    return df


def convert_data_to_examples(train, DATA_COLUMN, LABEL_COLUMN):
    train_InputExamples = train.apply(
        lambda x: InputExample(guid=None,
                               text_a=x[DATA_COLUMN],
                               text_b=None,
                               label=x[LABEL_COLUMN]), axis=1)

    return train_InputExamples


def convert_examples_to_tf_dataset(examples, tokenizer, max_length=512):
    features = []

    for e in examples:
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
                                                     input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int32),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    DATA_COLUMN = 'rapor'
    LABEL_COLUMN = 'sonuc'
    df_train = read_data('/home/hazal/nlp_dataset/brain_CT/example/train_labeled_data300_v4.xlsx')
    df_valid = read_data('/home/hazal/nlp_dataset/brain_CT/example/valid_labeled_data300_v4.xlsx')

    with tf.device('/cpu:0'):
        train_InputExamples, validation_InputExamples = convert_data_to_examples(df_train, df_valid, DATA_COLUMN,
                                                                                 LABEL_COLUMN)
        train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
        train_data = train_data.shuffle(100).batch(32).repeat(2)

        validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
        validation_data = validation_data.batch(32)
