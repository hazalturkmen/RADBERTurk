import argparse

import torch
import time
import datetime
from torch import cuda
from transformers import BertTokenizer

from pytorch.src.bert_layer import BERTClass
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')
device = 'cuda' if cuda.is_available() else 'cpu'
model = BERTClass()
model.to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
training_stats = []


def train(epoch, training_loader, valid_loader):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, EPOCHS))
    print('Training...')
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    t0 = time.time()

    model.train()
    for step, data in enumerate(training_loader, 0):

        if step % 40 == 0 and not step == 0:
            elapsed = format_time(time.time() - t0)

            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(training_loader), elapsed))

        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        outputs = model(ids, mask)
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)

        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)
        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    train_epoch_loss = tr_loss / nb_tr_steps
    train_epoch_accu = (n_correct * 100) / nb_tr_examples
    train_training_time = format_time(time.time() - t0)
    print(f"Training Loss Epoch: {train_epoch_loss}")
    print(f"Training Accuracy Epoch: {train_epoch_accu}")
    print("Training epcoh took: {:}".format(train_training_time))

    print("")
    print("Running Validation...")

    t0 = time.time()
    model.eval()
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0

    with torch.no_grad():
        for _, data in enumerate(valid_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

    val_epoch_loss = tr_loss / nb_tr_steps
    val_epoch_accu = (n_correct * 100) / nb_tr_examples
    val_validation_time = format_time(time.time() - t0)
    print(f"Validation Loss Epoch: {val_epoch_loss}")
    print(f"Validation Accuracy Epoch: {val_epoch_accu}")
    print("Validation took: {:}".format(val_validation_time))
    training_stats.append(
        {
            'epoch': epoch + 1,
            'Training Loss': train_epoch_loss,
            'Valid. Loss': val_epoch_loss,
            'Valid. Accur.': val_epoch_accu,
            'Training Time': train_training_time,
            'Validation Time': val_validation_time
        }
    )



def calcuate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct



def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


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

    for epoch in range(EPOCHS):
        train(epoch, training_loader, valid_loader)




