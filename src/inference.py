import re
import torch

import argparse
from model import DeepPunctuation, DeepPunctuationCRF
from config import *
from dataset import parse_data
from dataset_creation import split_words_and_punctuation, save_to_file, PUNCT_NAMES

def inference(deep_punctuation, tokenizer, args, device, token_style):
    deep_punctuation.eval()

    # create dataset
    with open(args.in_file, 'r', encoding='utf-8') as f:
        lines = [line for line in f.read().split('\n') if line.strip()]
    text = "\n".join(lines)
    text = re.sub(rf"[{''.join(PUNCT_NAMES.keys())}]", '', text)

    words_original_case, punctuation_mask = split_words_and_punctuation(text)
    save_to_file("data/pl/inference_temp", words_original_case, punctuation_mask, PUNCT_NAMES)

    data_items = parse_data("data/pl/inference_temp", tokenizer, args.sequence_length, token_style, is_inference=True)
    result = ""
    decode_idx = 0
    
    for x, _, attn_mask, y_mask in data_items:
        x = torch.tensor(x).reshape(1,-1)
        y_mask = torch.tensor(y_mask)
        attn_mask = torch.tensor(attn_mask).reshape(1,-1)
        x, attn_mask, y_mask = x.to(device), attn_mask.to(device), y_mask.to(device)

        with torch.no_grad():
            if args.use_crf:
                y = torch.zeros(x.shape[0])
                y_predict = deep_punctuation(x, attn_mask, y)
                y_predict = y_predict.view(-1)
            else:
                y_predict = deep_punctuation(x, attn_mask)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
        for i in range(y_mask.shape[0]):
            if y_mask[i] == 1:
                result += words_original_case[decode_idx] + punctuation_map[y_predict[i].item()] + ' '
                decode_idx += 1
                if decode_idx < len(words_original_case) and words_original_case[decode_idx] == '\n':
                    result += '\n'
                    decode_idx += 1

    print('Punctuated text:', result, sep="\n")
    with open(args.out_file, 'w', encoding='utf-8') as f:
        f.write(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Punctuation restoration inference on text file')
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
    parser.add_argument('--in-file', default='data/test_pl', type=str, help='path to inference file')
    parser.add_argument('--out-file', default='', type=str, help='output file location')

    parser.add_argument('--pretrained-model', default='xlm-roberta-large', type=str, help='pretrained language model')
    parser.add_argument('--weight-path', default='xlm-roberta-large.pt', type=str, help='model weight path')
    parser.add_argument('--lstm-dim', default=-1, type=int,
                        help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model')
    parser.add_argument('--use-crf', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='whether to use CRF layer or not')

    parser.add_argument('--run-id', default='58b6208c3f4c4d9eab209e82241e2a0b', type=str, help='run id from mlflow to get model from')
    parser.add_argument('--sequence-length', default=256, type=int,
                        help='sequence length to use when preparing dataset (default 256)')

    args = parser.parse_args()
    if args.run_id:
        import mlflow
        mlflow.set_tracking_uri('https://dagshub.com/annapanfil/punctuation_prediction.mlflow')
        print("Using DagsHub MLflow server https://dagshub.com/annapanfil/punctuation_prediction.mlflow")
    
        run = mlflow.get_run(args.run_id)
        args.pretrained_model = run.data.params["Base model"]
        args.use_crf = True if run.data.params["Use CRF"] == "True" else False

    if not args.out_file:
        args.out_file = args.in_file + '_out.txt'

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.pretrained_model]["tokenizer_name"])
    token_style = MODELS[args.pretrained_model]["token_style"]


    # Model
    device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')

    if args.run_id:
        model = mlflow.pytorch.load_model(f'runs:/{args.run_id}/models', map_location=device)
    
    else:
        model_save_path = args.weight_path

        if args.use_crf:
            model = DeepPunctuationCRF(args.pretrained_model, freeze_bert=False, lstm_dim=args.lstm_dim)
        else:
            model = DeepPunctuation(args.pretrained_model, freeze_bert=False, lstm_dim=args.lstm_dim)
        model.to(device)

        model.load_state_dict(torch.load(model_save_path))

    inference(model, tokenizer, args, device, token_style)
