import os
import torch
import numpy as np
import mlflow
from peft import LoraConfig, get_peft_model, TaskType
import seaborn as sns
from torch.utils import data
from torch.utils.data import random_split
import torch.nn as nn
import torch.multiprocessing
from tqdm import tqdm

from argparser import parse_arguments
from dataset import Dataset
from model import DeepPunctuation, DeepPunctuationWithPauses
from config import *
from test import test


def get_data_loaders(args, tokenizer):
    ### Get train, val and test dataloaders
    # In polish, when the test flag is set, the train and val sets are combined and split into 95% train and 5% val and test is used as test set
    # Otherwise, the train and val sets are separate and test is used as val set
    # @param args: arguments from argparser
    # @param tokenizer: tokenizer to use for tokenizing the text
    #
    # @return train_loader, val_loader, test_loaders: data loaders for train, val and test sets
    ###

    token_style = MODELS[args.pretrained_model]["token_style"]

    if args.language == 'english':
        train_set = Dataset(os.path.join(args.data_path, 'en/train2012'), 
                            tokenizer=tokenizer, 
                            sequence_len=args.sequence_length,
                            token_style=token_style,
                            use_durations=args.use_durations)
        val_set = Dataset(os.path.join(args.data_path, 'en/dev2012'), 
                          tokenizer=tokenizer, 
                          sequence_len=args.sequence_length,
                          token_style=token_style,
                          use_durations=args.use_durations)
        test_set_ref = Dataset(os.path.join(args.data_path, 'en/test2011'), 
                               tokenizer=tokenizer, 
                               sequence_len=args.sequence_length,
                               token_style=token_style,
                               use_durations=args.use_durations)
        test_set_asr = Dataset(os.path.join(args.data_path, 'en/test2011asr'), 
                               tokenizer=tokenizer, 
                               sequence_len=args.sequence_length,
                               token_style=token_style,
                               use_durations=args.use_durations)
        test_set = [val_set, test_set_ref, test_set_asr]

    elif args.language == 'polish':
        if args.test:
            trainval_set = Dataset(os.path.join(args.data_path, f'pl/trainval{"_" + args.data_variation if args.data_variation != "" else ""}'),
                                tokenizer=tokenizer,
                                sequence_len=args.sequence_length,
                                token_style=token_style,
                                use_durations=args.use_durations)

            # Split the dataset to have val set to make sure the model is learning well
            train_size = int(0.95 * len(trainval_set))  
            val_size = len(trainval_set) - train_size 
            train_set, val_set = random_split(trainval_set, [train_size, val_size])
            
            test_set = [Dataset(os.path.join(args.data_path, f'pl/test{"_" + args.data_variation if args.data_variation != "" else ""}'),
                            tokenizer=tokenizer,
                            sequence_len=args.sequence_length,
                            token_style=token_style,
                            use_durations=args.use_durations)]

        else:
            train_set = Dataset(os.path.join(args.data_path, f'pl/train{"_" + args.data_variation if args.data_variation != "" else ""}'),
                                tokenizer=tokenizer, 
                                sequence_len=args.sequence_length,
                                token_style=token_style,
                                use_durations=args.use_durations)
            val_set = Dataset(os.path.join(args.data_path, f'pl/val{"_" + args.data_variation if args.data_variation != "" else ""}'),
                                tokenizer=tokenizer,
                                sequence_len=args.sequence_length,
                                token_style=token_style,
                                use_durations=args.use_durations)
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

    return train_loader, val_loader, test_loaders

def train(args, deep_punctuation, device, train_loader, val_loader, test_loaders, criterion):
    ### Train the model and optionally log the results in mlflow
    # @param args: arguments from argparser
    # @param deep_punctuation: model to train
    # @param device: device to use for training
    # @param train_loader: data loader for training set
    # @param val_loader: data loader for validation set
    # @param test_loaders: data loaders for test sets
    # @param criterion: loss function
    #
    # @return: None
    ###

    best_val_score = 0

    with mlflow.start_run():
        if args.log:
            mlflow.set_tag('mlflow.runName', f'{args.pretrained_model}_{args.name}')
            mlflow.log_param("Base model", args.pretrained_model)
            mlflow.log_param("LSTM", args.lstm)
            mlflow.log_param("LSTM Dimension", args.lstm_dim)
            mlflow.log_param("Freeze bert", args.freeze_bert)
            mlflow.log_param("Epochs", args.epoch)
            mlflow.log_param("Learning Rate", args.lr)
            mlflow.log_param("Decay", args.decay)
            mlflow.log_param("Batch Size", args.batch_size)
            mlflow.log_param("Sequence Length", args.sequence_length)

        for epoch in range(args.epoch):
            train_loss = 0.0
            train_iteration = 0
            correct = 0
            total = 0
            best_model_state = deep_punctuation.state_dict()
            deep_punctuation.train()
            for x, y, att, y_mask, durations in tqdm(train_loader, desc='train'):
                x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
                y_mask = y_mask.view(-1)
                
                if args.use_durations:
                    durations = durations.to(device)
                    y_predict = deep_punctuation(x, att, pause_durations=durations)
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

            # metrics after each epoch
            train_loss /= train_iteration

            precision, recall, f1, _, _, val_score, val_loss = test(val_loader, deep_punctuation, device, args, "eval", criterion)
            _, _, _, _, _, train_score, _ = test(train_loader, deep_punctuation, device, args, "eval", criterion)
            
            print(f'epoch: {epoch}, Train loss: {train_loss}, Train score: {train_score}, Val loss: {val_loss}, Val score: {val_score}')
            if val_score > best_val_score:
                best_val_score = val_score
                best_model_state = deep_punctuation.state_dict()

            if args.log:
                mlflow.log_metric("Train score", train_score, step=epoch) # final scoring (weighted F1 score)
                mlflow.log_metric("Train Loss", train_loss, step=epoch)
                mlflow.log_metric("Validation score", val_score, step=epoch)
                mlflow.log_metric("Validation Loss", val_loss, step=epoch)
                if not args.test:
                    for punct, i in punctuation_dict.items():
                        mlflow.log_metric(f"Precision_{punct}", precision[i], step=epoch)
                        mlflow.log_metric(f"Recall_{punct}", recall[i], step=epoch)
                        mlflow.log_metric(f"F1_{punct}", f1[i], step=epoch)

        print('Best validation score:', best_val_score, "------------------------------")
        deep_punctuation.load_state_dict(best_model_state)
        if args.log:
            mlflow.pytorch.log_model(deep_punctuation, "models")

        # metrics for final model
        for loader in test_loaders:
            precision, recall, f1, cm, support, final_scoring, loss = test(loader, deep_punctuation, device, args)

            if args.log:
                for punct, i in punctuation_dict.items():
                    mlflow.log_metric(f"Precision_{punct}", precision[i])
                    mlflow.log_metric(f"Recall_{punct}", recall[i])
                    mlflow.log_metric(f"F1_{punct}", f1[i])

                mlflow.log_metric("Final Scoring", final_scoring, step=epoch)

                # plot 
                cm_img = sns.heatmap(cm, annot=True, cmap="coolwarm", xticklabels=punctuation_dict.keys(), yticklabels=punctuation_dict.keys(), fmt='d')
                cm_img.set_xlabel("correct")
                cm_img.set_ylabel("predicted")
                mlflow.log_figure(cm_img.get_figure(), "confusion_matrix.png")
                
            print(f"\t\t{punctuation_dict.values()} all_punctuation\nPrecision: {precision}\nRecall: {recall}\nF1 score: {f1}\nSupport: {support}")
            print("Final score:", final_scoring)
            print('Confusion Matrix', str(cm))


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')   # https://github.com/pytorch/pytorch/issues/11201

    args = parse_arguments()

    # for reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.pretrained_model]["tokenizer_name"])

    # data loaders
    train_loader, val_loader, test_loaders = get_data_loaders(args, tokenizer)
    print("Data loaders created")

    # Model
    device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
    if args.use_durations:
        model = DeepPunctuationWithPauses(args.pretrained_model, freeze_bert=args.freeze_bert, lstm=args.lstm, lstm_dim=args.lstm_dim)
    else:
        model = DeepPunctuation(args.pretrained_model, freeze_bert=args.freeze_bert, lstm=args.lstm, lstm_dim=args.lstm_dim)
    
    print(model)
    model.to(device)

    if args.use_lora:
        lora_config = LoraConfig(
            r=128,
            lora_alpha=256,
            target_modules=["key", "query", "value", "dense"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.TOKEN_CLS
        )

        model = get_peft_model(model, lora_config)

        model.print_trainable_parameters()

    criterion = nn.CrossEntropyLoss()

    # if args.use_durations:
    #     ll_params = model.linear_layer.parameters()
    #     params_except_linear = [param for name, param in model.named_parameters() if 'linear_layer' not in name]

    #     optimizer = torch.optim.Adam([
    #                 {'params': params_except_linear, 'lr': args.lr},
    #                 {'params': ll_params, 'lr': 0.001}
    #             ], weight_decay=args.decay)
    # else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    print("Model created")

    # setup mlflow server
    if args.log:
        try:
            _ = os.environ['MLFLOW_TRACKING_USERNAME']
            _ = os.environ['MLFLOW_TRACKING_PASSWORD']
            mlflow.set_tracking_uri('https://dagshub.com/annapanfil/punctuation_prediction.mlflow')
            print("Using DagsHub MLflow server https://dagshub.com/annapanfil/punctuation_prediction.mlflow")
        except KeyError:
            mlflow.set_tracking_uri("http://localhost:8080")
            print("Using local MLflow server http://localhost:8080.\nMake sure you have mlflow server running (mlflow server --host 127.0.0.1 --port 8080).")
        
        mlflow.set_experiment(args.experiment_name)

    print(f"training {args.pretrained_model}_{args.data_variation}_{args.name}")
    train(args, model, device, train_loader, val_loader, test_loaders, criterion)
