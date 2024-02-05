from torch import nn
from transformers import ConvNextV2Model
from transformers.models.convnextv2.modeling_convnextv2 import ConvNextV2LayerNorm, ConvNextV2GRN


class ConvNextV2ForMultiLabelClassification(nn.Module):
    def __init__(self, config, output_size, vocab_size, use_sigmoid):
        super().__init__()
        self.body = ConvNextV2Model(config)
        self.head = nn.Linear(output_size, vocab_size)
        if use_sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x):
        x = self.body(x)
        x = x.pooler_output
        x = self.head(x)
        x = self.activation(x)
        return x
    
    def get_optimizer_groups(self, weight_decay):
        # https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv2d)
        blacklist_weight_modules = (nn.LayerNorm, ConvNextV2LayerNorm, ConvNextV2GRN)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups
