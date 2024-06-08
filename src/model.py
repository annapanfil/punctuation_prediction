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
    
    
class DeepPunctuationWithPauses(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm="bi", lstm_dim=-1):
        super().__init__()
        self.output_dim = len(punctuation_dict)
        self.has_lstm = lstm != "none"

        # BERT layers
        self.bert = AutoModel.from_pretrained(MODELS[pretrained_model]["model_name"])
        self.config = self.bert.config 

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        
        bert_dim = MODELS[pretrained_model]["output_dimension"]

        self.linear_layer = nn.Linear(self.bert.config.hidden_size + 1, self.bert.config.hidden_size)
    
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
                    output_attentions=None, output_hidden_states=None, return_dict=None, labels=None, pause_durations=None):
        
        if len(input_ids.shape) == 1:
            input_ids = input_ids.view(1, input_ids.shape[0])  # add dummy batch for single sample
        
        input_shape = input_ids.size()

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)

    

        # BERT 
        token_embeddings = self.bert.embeddings.word_embeddings(input_ids)
        position_ids = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(input_ids.size())
        position_embeddings = self.bert.embeddings.position_embeddings(position_ids)
        token_type_embeddings = self.bert.embeddings.token_type_embeddings(token_type_ids)
        
        pause_durations = pause_durations.unsqueeze(-1).float()  # Ensure it's float and of shape (batch_size, seq_len, 1)
        
        # Combine all embeddings
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = torch.cat((embeddings, pause_durations), dim=-1)
        embeddings = self.linear_layer(embeddings)


        extended_attention_mask = self.bert.get_extended_attention_mask(attention_mask, input_ids.shape, input_ids.device)
        head_mask = self.bert.get_head_mask(None, self.bert.config.num_hidden_layers)
        
        encoder_outputs = self.bert.encoder(
            embeddings,
            attention_mask=extended_attention_mask,
            head_mask=head_mask
        )
        
        output = encoder_outputs[0]
        
        if self.has_lstm:
            output = torch.transpose(output, 0, 1)
            output, (_, _) = self.lstm(output)
        
        output = torch.transpose(output, 0, 1)
        output = self.linear(output)
        
        return output