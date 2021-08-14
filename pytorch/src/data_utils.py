import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import seaborn as sns

from dataset.labeled_data import Radataset

MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16


def load_data(train_xlsx_path, dev_xlsx_path, test_xlsx_path, tokenizer):
    tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
    training_set = Radataset(train_xlsx_path, tokenizer, MAX_LEN)
    valid_set = Radataset(dev_xlsx_path, tokenizer, MAX_LEN)
    testing_set = Radataset(test_xlsx_path, tokenizer, MAX_LEN)

    print("TRAIN Dataset: {}".format(training_set.data.shape))
    print("VALID Dataset: {}".format(valid_set.data.shape))
    print("TEST Dataset: {}".format(testing_set.data.shape))

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    valid_loader = DataLoader(valid_set, **train_params)
    testing_loader = DataLoader(testing_set, **train_params)

    return training_loader, valid_loader, testing_loader


def write(training_stats):
    pd.set_option('precision', 2)

    # Create a DataFrame from our training statistics.
    df_stats = pd.DataFrame(data=training_stats)

    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index('epoch')
    df_stats.to_csv('stats.csv')
    return df_stats


def plot_loss(df_stats):
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')


    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    plt.show()

    fig1 = plt.figure()
    fig1.savefig('fig1.png')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    #plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    fig2 = plt.figure()
    fig2.savefig('fig2.png')
