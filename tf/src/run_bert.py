import argparse
from transformers import TFBertForSequenceClassification
from data_utils import convert_data_to_examples
from data_utils import read_data
from data_utils import convert_examples_to_tf_dataset
from transformers import BertTokenizer
import tensorflow as tf
import time


def create_model():
    model = TFBertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased", num_labels=3)
    model.summary()
    return model


def fine_tune(model, x_train, x_val):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=tf.metrics.SparseCategoricalAccuracy(),
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath= '/tmp/checkpoint', monitor='sparse_categorical_accuracy', verbose=1,
                     save_best_only=True, mode='max',save_freq='epoch',shuffle=True,save_weights_only=True)
    model.fit(x_train, epochs=5, validation_data=x_val, callbacks=checkpoint)



def evaluate_model(model, test_data):
    loss, accuracy = model.evaluate(test_data)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BERT-base model on task of labeling radiology reports.')

    parser.add_argument('--train_xlsx', type=str, nargs='?', required=True,
                        help='path to xlsx containing train reports.')
    parser.add_argument('--dev_xlsx', type=str, nargs='?', required=True,
                        help='path to xlsx containing dev reports.')
    parser.add_argument('--test_xlsx', type=str, nargs='?', required=True,
                        help='path to xlsx containing test reports.')

    """parser.add_argument('--output_dir', type=str, nargs='?', required=True,
                        help='path to output directory where checkpoints will be saved')"""

    args = parser.parse_args()

    train_xlsx_path = args.train_xlsx
    dev_xlsx_path = args.dev_xlsx
    test_xlsx_path = args.test_xlsx
    tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    DATA_COLUMN = 'rapor'
    LABEL_COLUMN = 'sonuc'
    df_train = read_data(train_xlsx_path)
    df_valid = read_data(dev_xlsx_path)
    df_test = read_data(test_xlsx_path)
    with tf.device('/cpu:0'):
        train_InputExamples = convert_data_to_examples(df_train, DATA_COLUMN, LABEL_COLUMN)
        validation_InputExamples = convert_data_to_examples(df_valid, DATA_COLUMN, LABEL_COLUMN)
        test_InputExamples = convert_data_to_examples(df_test, DATA_COLUMN, LABEL_COLUMN)

        train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
        train_data = train_data.shuffle(100).batch(32).repeat(2)

        validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
        validation_data = validation_data.batch(32)

        test_data = convert_examples_to_tf_dataset(list(test_InputExamples), tokenizer)
        test_data = test_data.shuffle(100).batch(32).repeat(2)

    model = create_model()
    print('--Training is started---')
    begin_time = time.time()
    fine_tune(model, train_data, validation_data)
    print('Seconds since model training= ',(time.time()-begin_time))
    evaluate_model(model,test_data)
