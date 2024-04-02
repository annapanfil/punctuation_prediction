import os
import torch
import numpy as np
import mlflow
import seaborn as sns
from torch.utils import data
import torch.nn as nn
import torch.multiprocessing
from tqdm import tqdm

from argparser import parse_arguments
from dataset import Dataset
import datetime
from model import DeepPunctuation, DeepPunctuationCRF
from config import *
import augmentation


torch.multiprocessing.set_sharing_strategy('file_system')   # https://github.com/pytorch/pytorch/issues/11201

args = parse_arguments()

ml_log = args.log

# for reproducibility
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODELS[args.pretrained_model]["tokenizer_name"])
augmentation.tokenizer = tokenizer
augmentation.sub_style = args.sub_style
augmentation.alpha_sub = args.alpha_sub
augmentation.alpha_del = args.alpha_del
token_style = MODELS[args.pretrained_model]["token_style"]
ar = args.augment_rate
sequence_len = args.sequence_length
aug_type = args.augment_type

# Datasets
if args.language == 'english':
    train_set = Dataset(os.path.join(args.data_path, 'en/train2012'), tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type)
    val_set = Dataset(os.path.join(args.data_path, 'en/dev2012'), tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False)
    test_set_ref = Dataset(os.path.join(args.data_path, 'en/test2011'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False)
    test_set_asr = Dataset(os.path.join(args.data_path, 'en/test2011asr'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False)
    test_set = [val_set, test_set_ref, test_set_asr]
elif args.language == 'polish':
    train_set = Dataset(os.path.join(args.data_path, 'pl/train'), tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type)
    val_set = Dataset(os.path.join(args.data_path, 'pl/val'), tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=False)
    test_set = [val_set]
else:
    raise ValueError('Incorrect language argument for Dataset')

# Data Loaders
data_loader_params = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': 1
}
train_loader = torch.utils.data.DataLoader(train_set, **data_loader_params)
val_loader = torch.utils.data.DataLoader(val_set, **data_loader_params)
test_loaders = [torch.utils.data.DataLoader(x, **data_loader_params) for x in test_set]

# Model
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
if args.use_crf:
    deep_punctuation = DeepPunctuationCRF(args.pretrained_model, freeze_bert=args.freeze_bert, lstm_dim=args.lstm_dim)
else:
    deep_punctuation = DeepPunctuation(args.pretrained_model, freeze_bert=args.freeze_bert, lstm_dim=args.lstm_dim)
deep_punctuation.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(deep_punctuation.parameters(), lr=args.lr, weight_decay=args.decay)

def validate(data_loader):
    """
    :return: validation accuracy, validation loss
    """
    num_iteration = 0
    deep_punctuation.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for x, y, att, y_mask in tqdm(data_loader, desc='eval'):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            if args.use_crf:
                y_predict = deep_punctuation(x, att, y)
                loss = deep_punctuation.log_likelihood(x, att, y)
                y_predict = y_predict.view(-1)
                y = y.view(-1)
            else:
                y_predict = deep_punctuation(x, att)
                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                loss = criterion(y_predict, y)
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
            val_loss += loss.item()
            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()
    return correct/total, val_loss/num_iteration


def test(data_loader):
    """
    :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy, confusion matrix
    """
    num_iteration = 0
    deep_punctuation.eval()
    # +1 for overall result
    tp = np.zeros(1+len(punctuation_dict), dtype=int)
    fp = np.zeros(1+len(punctuation_dict), dtype=int)
    fn = np.zeros(1+len(punctuation_dict), dtype=int)
    cm = np.zeros((len(punctuation_dict), len(punctuation_dict)), dtype=int)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, att, y_mask in tqdm(data_loader, desc='test'):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            if args.use_crf:
                y_predict = deep_punctuation(x, att, y)
                y_predict = y_predict.view(-1)
                y = y.view(-1)
            else:
                y_predict = deep_punctuation(x, att)
                y = y.view(-1)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()
            for i in range(y.shape[0]):
                if y_mask[i] == 0:
                    # we can ignore this because we know there won't be any punctuation in this position
                    # since we created this position due to padding or sub-word tokenization
                    continue
                cor = y[i]
                prd = y_predict[i]
                if cor == prd:
                    tp[cor] += 1
                else:
                    fn[cor] += 1
                    fp[prd] += 1
                cm[cor][prd] += 1
    # ignore first index which is for no punctuation
    tp[-1] = np.sum(tp[1:])
    fp[-1] = np.sum(fp[1:])
    fn[-1] = np.sum(fn[1:])
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, correct/total, cm


def train():
    best_val_acc = 0

    try:
        _ = os.environ['MLFLOW_TRACKING_USERNAME']
        _ = os.environ['MLFLOW_TRACKING_PASSWORD']
        mlflow.set_tracking_uri('https://dagshub.com/annapanfil/punctuation_prediction.mlflow')
        print("Using DagsHub MLflow server https://dagshub.com/annapanfil/punctuation_prediction.mlflow")
    except KeyError:
        mlflow.set_tracking_uri("http://localhost:5000")
        print("Using local MLflow server http://localhost:5000")
        
    
    mlflow.set_experiment("PunctuationPrediction")

    with mlflow.start_run():
        if ml_log:
            mlflow.set_tag('mlflow.runName', f'{args.language}_{datetime.datetime.now().strftime("%Y.%m.%d_%H:%M:%S")}')
            mlflow.log_param("Number of Epochs", args.epoch)
            mlflow.log_param("Learning Rate", args.lr)
            mlflow.log_param("Batch Size", args.batch_size)
            mlflow.log_param("Language", args.language)
            mlflow.log_param("Decay", args.decay)
            mlflow.log_param("Use CRF", args.use_crf)

        for epoch in range(args.epoch):
            train_loss = 0.0
            train_iteration = 0
            correct = 0
            total = 0
            best_model_state = deep_punctuation.state_dict()
            deep_punctuation.train()
            for x, y, att, y_mask in tqdm(train_loader, desc='train'):
                x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
                y_mask = y_mask.view(-1)
                if args.use_crf:
                    loss = deep_punctuation.log_likelihood(x, att, y)
                    # y_predict = deep_punctuation(x, att, y)
                    # y_predict = y_predict.view(-1)
                    y = y.view(-1)
                else:
                    y_predict = deep_punctuation(x, att)
                    y_predict = y_predict.view(-1, y_predict.shape[2])
                    y = y.view(-1)
                    loss = criterion(y_predict, y)
                    y_predict = torch.argmax(y_predict, dim=1).view(-1)

                    correct += torch.sum(y_mask * (y_predict == y).long()).item()

                optimizer.zero_grad()
                train_loss += loss.item()
                train_iteration += 1
                loss.backward()

                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(deep_punctuation.parameters(), args.gradient_clip)
                optimizer.step()

                y_mask = y_mask.view(-1)

                total += torch.sum(y_mask).item()

            train_loss /= train_iteration
            print('epoch: {}, Train loss: {}, Train accuracy: {}'.format(epoch, train_loss, correct / total))

            val_acc, val_loss = validate(val_loader)

            print('epoch: {}, Val loss: {}, Val accuracy: {}'.format(epoch, val_loss, val_acc))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = deep_punctuation.state_dict()

            if ml_log:
                mlflow.log_metric("Validation Accuracy", val_acc, step=epoch)
                mlflow.log_metric("Validation Loss", val_loss, step=epoch)


        print('Best validation Acc:', best_val_acc)
        deep_punctuation.load_state_dict(best_model_state)
        if ml_log:
            mlflow.pytorch.log_model(deep_punctuation, "models")

        for loader in test_loaders:
            precision, recall, f1, accuracy, cm = test(loader)
            print('Precision: ' + str(precision) + '\n' + 'Recall: ' + str(recall) + '\n' + 'F1 score: ' + str(f1) + \
                '\n' + 'Accuracy:' + str(accuracy) + '\n' + 'Confusion Matrix' + str(cm) + '\n')

            # Po każdym teście w pętli treningowej, dodaj to:

            if ml_log:
                mlflow.log_metric("Test Accuracy", accuracy, step=epoch)
                mlflow.log_metric("Test F1-Score", np.nanmean(f1), step=epoch)
                mlflow.log_metric("Test Precision", np.nanmean(precision), step=epoch)
                mlflow.log_metric("Test Recall", np.nanmean(recall), step=epoch)
                # plot 
                cm_img = sns.heatmap(cm, annot=True, cmap="coolwarm", xticklabels=punctuation_dict.keys(), yticklabels=punctuation_dict.keys())
                cm_img.set_xlabel("correct")
                cm_img.set_ylabel("predicted")
                mlflow.log_figure(cm_img.get_figure(), "confusion_matrix.png")
                


if __name__ == '__main__':
    train()
