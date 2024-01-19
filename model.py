import torch
from transformers import ConvNextV2Model


class ConvNextV2ForMultiLabelClassification(torch.nn.Module):
    def __init__(self, config, output_size, vocab_size, use_sigmoid):
        super().__init__()
        self.body = ConvNextV2Model(config)
        self.head = torch.nn.Linear(output_size, vocab_size)
        if use_sigmoid:
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = torch.nn.Identity()
    
    def forward(self, x):
        x = self.body(x)
        x = x.pooler_output
        x = self.head(x)
        x = self.activation(x)
        return x
