import argparse
import os
import time
import torch
import datetime
from matplotlib import pyplot as plt
from torch import cuda
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
from bert_layer import BertClass
from data_utils import plot_confusion_matrix, load_data, write, plot_loss
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
from config import process_config

training_stats = []


def initialize_model(epochs, train_data, lr):
    """Initialize the Bert Classifier, the optimizer the learning rate scheduler and the loss function.
    """

    bert_classifier = BertClass(freeze_bert=False)

    # to utilize multiple GPU's
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        bert_classifier = nn.DataParallel(bert_classifier)
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=lr,
                      eps=1e-8  # Default epsilon value
                      )

    # Total number of training steps
    total_steps = len(train_data) * epochs

    # Loss function for multiclass text classifier
    loss_function = torch.nn.CrossEntropyLoss()

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value
                                                num_training_steps=total_steps)
    if "checkpoints":
        model_state, optimizer_state = torch.load(
            os.path.join("checkpoints", "checkpoint"))
        bert_classifier.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    return bert_classifier, optimizer, scheduler, loss_function


def train(save_path, epoch, training_loader, valid_loader, model, optimizer, scheduler, total_epoch):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, total_epoch))
    print('Training...')
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    t0 = time.time()
    best_metric = 0.0

    model.train()
    for step, data in enumerate(training_loader, 0):

        if (step % 10 == 0 and not step == 0) or (step == len(training_loader) - 1):
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
        n_correct += calculate_accu(big_idx, targets)
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)
        optimizer.zero_grad()
        loss.backward()
        # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    train_epoch_loss = tr_loss / nb_tr_steps
    train_epoch_accu = (n_correct * 100) / nb_tr_examples
    train_training_time = format_time(time.time() - t0)
    print(f"Training Loss Epoch: {train_epoch_loss}")
    print(f"Training Accuracy Epoch: {train_epoch_accu}")
    print("Training epcoh took: {:}".format(train_training_time))

    print("")
    print("Running Validation...")
    time.gmtime(0)
    t1 = time.time()
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
            n_correct += calculate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

    val_epoch_loss = tr_loss / nb_tr_steps
    val_epoch_accu = (n_correct * 100) / nb_tr_examples
    val_validation_time = format_time(time.time() - t1)
    print(f"Validation Loss Epoch: {val_epoch_loss}")
    print(f"Validation Accuracy Epoch: {val_epoch_accu}")
    print("Validation took: {:}".format(val_validation_time))
    metric_avg = val_epoch_accu

    # save the best model
    if metric_avg > best_metric:
        print("saving new best network!\n")
        best_metric = metric_avg
        path = os.path.join(save_path, "model_epoch%d" % (epoch + 1))
        torch.save({'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   path)

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


def calculate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def evaluate(model, testing_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            output = model(ids, mask)
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(targets.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1, 0, 2]))

    cnf_matrix = confusion_matrix(y_true, y_pred, labels=[1, 0, 2])
    np.set_printoptions(precision=2)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['VAR', 'YOK', 'SERİDIŞI'],
                          title='Confusion matrix, without normalization')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BERT-base model on task of labeling radiology reports.')

    parser.add_argument('--train_xlsx', type=str, nargs='?', required=True,
                        help='path to xlsx containing train reports.')
    parser.add_argument('--dev_xlsx', type=str, nargs='?', required=True,
                        help='path to xlsx containing dev reports.')
    parser.add_argument('--test_xlsx', type=str, nargs='?', required=True,
                        help='path to xlsx containing test reports.')
    parser.add_argument('--output_dir', type=str, nargs='?', required=True,
                        help='path to output directory where checkpoints will be saved')
    parser.add_argument('--config_json', type=str, nargs='?', required=True,
                        help='path to config containing training parameters')

    args = parser.parse_args()
    train_xlsx_path = args.train_xlsx
    dev_xlsx_path = args.dev_xlsx
    test_xlsx_path = args.test_xlsx
    out_path = args.output_dir
    config_path = args.config_json

    config = process_config(config_path)
    training_loader, valid_loader, test_loader = load_data(train_xlsx_path, dev_xlsx_path, test_xlsx_path, config)
    # device
    device = 'cuda' if cuda.is_available() else 'cpu'
    # initialize model
    bert_classifier, optimizer, scheduler, loss_function = initialize_model(config.num_epochs, training_loader,
                                                                            config.learning_rate)
    # start total training time
    total = time.time()

    # Training loop
    for epoch in range(config.num_epochs):
        train(out_path, epoch, training_loader, valid_loader, bert_classifier, optimizer, scheduler, config.num_epochs)
    # Evaluate model
    evaluate(bert_classifier, test_loader)
    # Calculate total training time
    total_train_training_time = format_time(time.time() - total)
    print("----------------")
    print()
    print("Total Training time: {:}".format(total_train_training_time))
    print()
    # print stats and loss
    df_stats = write(training_stats)
    plot_loss(df_stats)
