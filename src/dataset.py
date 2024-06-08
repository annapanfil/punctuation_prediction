import torch
from config import *
import numpy as np


def parse_data(file_path, tokenizer, sequence_len, token_style, use_durations=False, is_inference=False):
    """
    :param file_path: text file path that contains tokens and punctuations separated by tab in lines.
                      It can also contain new lines in the end of each document
    :param tokenizer: tokenizer that will be used to further tokenize word for BERT like models
    :param sequence_len: maximum length of each sequence
    :param token_style: For getting index of special tokens in config.TOKEN_IDX
    :param use_durations: whether to use pause durations. If yes, they are taken from file_path + "_durations"
    :return: list of [tokens_index, punctuation_index, attention_masks, punctuation_mask], each having sequence_len
    punctuation_mask is used to ignore special indices like padding and intermediate sub-word token during evaluation
    """
    data_items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f.read().split('\n')]

        if use_durations:
            dur_f = open(file_path + "_durations", 'r', encoding='utf-8')
            dur_lines = [line for line in dur_f.read().split('\n')]
            assert len(lines) == len(dur_lines), f"Number of lines in text and duration files should be same ({len(lines)} != {len(dur_lines)})"

        idx = 0
        # loop until end of the entire text
        while idx < len(lines):
            # add beginning of sequence token
            x = [TOKEN_IDX[token_style]['START_SEQ']]
            y = [0]
            y_mask = [0] if is_inference else [1]  # which positions we need to consider while evaluating i.e., ignore pad or sub tokens
            durations = [0]

            # loop through words until we have required sequence length or new line is encountered
            # -1 because we will have a special end of sequence token at the end
            while len(x) < sequence_len - 1 and idx < len(lines):
                if lines[idx] == "":
                    idx += 1
                    break

                word, punc = lines[idx].split('\t')
                tokens = tokenizer.tokenize(word)
                # if taking these tokens exceeds sequence length we finish current sequence with padding
                # then start next sequence from this token
                if len(x) + len(tokens) >= sequence_len:
                    break
                else:
                    # add middle tokend with mask as 0
                    for i in range(len(tokens) - 1):
                        x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                        y.append(0)
                        y_mask.append(0)
                        durations.append(0)
                    # add last token with mask as 1 and punctuation
                    if len(tokens) > 0:
                        x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                    else:
                        x.append(TOKEN_IDX[token_style]['UNK'])
                    y.append(punctuation_dict[punc])
                    y_mask.append(1)
                    if use_durations:
                        durations.append(float(dur_lines[idx]))
                    idx += 1
            x.append(TOKEN_IDX[token_style]['END_SEQ'])
            y.append(0)
            if not is_inference:
                y_mask.append(1)
            else:
                y_mask.append(0)
            durations.append(0)

            # add padding if sequence length is not reached
            if len(x) < sequence_len:
                x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
                y = y + [0 for _ in range(sequence_len - len(y))]
                y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
                durations = durations + [0 for _ in range(sequence_len - len(durations))]
            attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]

            if use_durations:
                assert len(durations) == len(x), f"Length of durations and tokens should be same ({len(durations)} != {len(x)})"

            if not use_durations:
                durations = [-1 for _ in x]
            data_items.append([x, y, attn_mask, y_mask, durations])

        if use_durations:
            dur_f.close()

    return data_items


class Dataset(torch.utils.data.Dataset):
    def __init__(self, files, tokenizer, sequence_len, token_style, use_durations=False):
        """

        :param files: single file or list of text files containing tokens and punctuations separated by tab in lines
        :param tokenizer: tokenizer that will be used to further tokenize word for BERT like models
        :param sequence_len: length of each sequence
        :param token_style: For getting index of special tokens in config.TOKEN_IDX
        :param use_durations: whether to use pause durations or not
        """
        self.use_durations = use_durations

        if isinstance(files, list):
            self.data = []
            for file in files:
                self.data += parse_data(file, tokenizer, sequence_len, token_style, use_durations)
        else:
            self.data = parse_data(files, tokenizer, sequence_len, token_style, use_durations)
        self.sequence_len = sequence_len
        self.token_style = token_style

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        attn_mask = self.data[index][2]
        y_mask = self.data[index][3]
        durations = torch.tensor(self.data[index][4]) # filled with -1 if not using durations
        
        x = torch.tensor(x)
        y = torch.tensor(y)
        attn_mask = torch.tensor(attn_mask)
        y_mask = torch.tensor(y_mask)

        return x, y, attn_mask, y_mask, durations


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODELS["herbert-base"]["tokenizer_name"])
    data_items = parse_data("data/debug_data", tokenizer, 15, MODELS["herbert-base"]["token_style"], use_durations=False)
    print(data_items)
    for i in range(len(data_items)):
        print(tokenizer.decode(data_items[i][0]))
