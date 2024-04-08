from transformers import *

# special tokens indices in different models available in transformers
TOKEN_IDX = {
    'bert': {
        'START_SEQ': 101,
        'PAD': 0,
        'END_SEQ': 102,
        'UNK': 100
    },
    'xlm': {
        'START_SEQ': 0,
        'PAD': 2,
        'END_SEQ': 1,
        'UNK': 3
    },
    'roberta': {
        'START_SEQ': 0,
        'PAD': 1,
        'END_SEQ': 2,
        'UNK': 3
    },
    'albert': {
        'START_SEQ': 2,
        'PAD': 0,
        'END_SEQ': 3,
        'UNK': 1
    },
}

# 'O' -> No punctuation
punctuation_dict = {
    'O': 0, 
    'COMMA': 1, 
    'PERIOD': 2, 
    'QUESTION': 3, 
    'EXCLAMATION': 4, 
    "DASH": 5, 
    "COLON": 6, 
    "ELLIPSIS": 7}


# pretrained model name: (model_name, tokenizer_name, output dimension, token style)
MODELS = {
    'bert-base-uncased': {"model_name": 'bert-base-uncased', 
                          "tokenizer_name": 'bert-base-uncased', 
                          "output_dimension": 768, 
                          "token_style": 'bert'},
    'herbert-base':  {"model_name": 'allegro/herbert-klej-cased-v1', 
                          "tokenizer_name": 'allegro/herbert-klej-cased-tokenizer-v1', 
                          "output_dimension": 768, 
                          "token_style": 'xlm'},
    'bert-multiling-uncased': {"model_name": "bert-base-multilingual-uncased",
                               "tokenizer_name": "bert-base-multilingual-uncased",
                               "output_dimension": 768,
                               "token_style": 'bert'},
    'roberta-multiling': {"model_name": "xlm-roberta-base",
                          "tokenizer_name": "xlm-roberta-base",
                          "output_dimension": 768,
                          "token_style": 'roberta'},
    'polish-roberta': {"model_name": "dkleczek/Polish_RoBERTa_large_OPI",
                        "tokenizer_name": "dkleczek/Polish_RoBERTa_large_OPI",
                        "output_dimension": 1024,
                        "token_style": 'roberta'},


}
