import torch.nn as nn
import torch
from config import *

class DeepPunctuation(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm="bi", lstm_dim=-1):
        super(DeepPunctuation, self).__init__()
        self.output_dim = len(punctuation_dict)
        self.has_lstm = lstm != "none"
        self.bert_layer = AutoModel.from_pretrained(MODELS[pretrained_model]["model_name"])
        self.config = self.bert_layer.config 
        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        bert_dim = MODELS[pretrained_model]["output_dimension"]
        if lstm_dim == -1 or lstm == "none":
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim

        if lstm != "none":
            bidirectional = True if lstm == "bi" else False
            self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=hidden_size, num_layers=1, bidirectional=bidirectional)
        in_features = hidden_size*2 if lstm == "bi" else hidden_size
        self.linear = nn.Linear(in_features=in_features, out_features=len(punctuation_dict))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, inputs_embeds=None,
                 output_attentions=None, output_hidden_states=None, return_dict=None, labels=None):
        if len(input_ids.shape) == 1:
            input_ids = input_ids.view(1, input_ids.shape[0])  # add dummy batch for single sample
        # (B, N, E) -> (B, N, E)
        output = self.bert_layer(input_ids, attention_mask=attention_mask, 
                            token_type_ids=token_type_ids, 
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions, 
                            output_hidden_states=output_hidden_states, 
                            return_dict=return_dict)[0]
        # (B, N, E) -> (N, B, E)
        output = torch.transpose(output, 0, 1)
        if self.has_lstm:
            output, (_, _) = self.lstm(output)
        # (N, B, E) -> (B, N, E)
        output = torch.transpose(output, 0, 1)
        output = self.linear(output)
        return output