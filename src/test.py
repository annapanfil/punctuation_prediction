import os
import torch
from tqdm import tqdm
import numpy as np

import argparse
from dataset import Dataset
from model import DeepPunctuation, DeepPunctuationCRF
from config import *


def test(data_loader, deep_punctuation, device, args, desc="test", criterion=torch.nn.CrossEntropyLoss()):
    """
    :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy, confusion matrix
    length of arrays is 1 + n_punctuation, in format e.g. 
    [1. → for no punct   0.1 → for period    0.02325581 → for comma    0.04651163 → for question mark    0.04166667 → for all punctuation signs (excluding no punctuation)]
    """
    num_iteration = 0
    deep_punctuation.eval()
    # +1 for overall result
    tp = np.zeros(1+len(punctuation_dict), dtype=int)
    fp = np.zeros(1+len(punctuation_dict), dtype=int)
    fn = np.zeros(1+len(punctuation_dict), dtype=int)
    cm = np.zeros((len(punctuation_dict), len(punctuation_dict)), dtype=int)
    support = np.zeros(len(punctuation_dict), dtype=int)
    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        for x, y, att, y_mask in tqdm(data_loader, desc=desc):
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
                support[cor] += 1
    # ignore first index which is for no punctuation
    tp[-1] = np.sum(tp[1:])
    fp[-1] = np.sum(fp[1:])
    fn[-1] = np.sum(fn[1:])
    precision = np.nan_to_num(tp/(tp+fp))
    recall = np.nan_to_num(tp/(tp+fn))
    f1 = np.nan_to_num(2 * precision * recall / (precision + recall))

    final_scoring = 0
    for punct, i in list(punctuation_dict.items())[1:]: # skip no punctuation
        final_scoring += np.nan_to_num(support[i] * f1[i])
        
    final_scoring /= sum(support[1:])

    return precision, recall, f1, cm, support, final_scoring, loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Punctuation restoration test')
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')

    parser.add_argument('--data-path', default='data/pl/val', type=str, help='path to test dataset')

    parser.add_argument('--run-id', default='', type=str, help='run id from mlflow to get model from')
    parser.add_argument('--lstm', default="bi", type=str, help='lstm type (bi, uni or none)')
    parser.add_argument('--lstm-dim', default=-1, type=int,
                        help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model')
    parser.add_argument('--sequence-length', default=256, type=int,
                        help='sequence length to use when preparing dataset (default 256)')

    parser.add_argument('--weight-path', default='out/weights.pt', type=str, help='model weight path')
    parser.add_argument('--pretrained-model', default='roberta-large', type=str, help='pretrained language model')
    parser.add_argument('--use-crf', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use CRF layer or not')
    parser.add_argument('--batch-size', default=8, type=int, help='batch size (default: 8)')

    parser.add_argument('--visualize-cm', default=True, type=lambda x: (str(x).lower() == 'true'), help="visualize confusion matrix")
    
    args = parser.parse_args()
    
    if args.run_id:
        import mlflow
        mlflow.set_tracking_uri('https://dagshub.com/annapanfil/punctuation_prediction.mlflow')
        print("Using DagsHub MLflow server https://dagshub.com/annapanfil/punctuation_prediction.mlflow")
    
        run = mlflow.get_run(args.run_id)
        args.pretrained_model = run.data.params["Base model"]
        args.batch_size = int(run.data.params["Batch Size"])
        args.use_crf = True if run.data.params["Use CRF"] == "True" else False

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.pretrained_model]["tokenizer_name"])
    token_style = MODELS[args.pretrained_model]["token_style"]

    
    test_set = Dataset(args.data_path, tokenizer=tokenizer, sequence_len=args.sequence_length,
                                token_style=token_style, is_train=False)

    # Data Loaders
    data_loader_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 0
    }

    test_loader = torch.utils.data.DataLoader(test_set, **data_loader_params)


    # Model
    device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')

    if args.run_id:
        model = mlflow.pytorch.load_model(f'runs:/{args.run_id}/models',  map_location=device)
    
    else:
        model_save_path = args.weight_path

        if args.use_crf:
            model = DeepPunctuationCRF(args.pretrained_model, freeze_bert=False, lstm_dim=args.lstm_dim)
        else:
            model = DeepPunctuation(args.pretrained_model, freeze_bert=False, lstm=args.lstm, lstm_dim=args.lstm_dim)
        model.to(device)

        model.load_state_dict(torch.load(model_save_path))


    precision, recall, f1, cm, support, final_scoring, loss = test(test_loader, model, device, args)

    print(f"\t\t{punctuation_dict.values()} all_punctuation\nPrecision: {precision}\nRecall: {recall}\nF1 score: {f1}")
    print("Final score:", final_scoring)
    print('Confusion Matrix', str(cm))

    if args.visualize_cm:
        import seaborn as sns
        import matplotlib.pyplot as plt
        cm_img = sns.heatmap(cm, annot=True, cmap="coolwarm", xticklabels=punctuation_dict.keys(), yticklabels=punctuation_dict.keys(), fmt='d')
        cm_img.set_xlabel("correct")
        cm_img.set_ylabel("predicted")
        plt.show()