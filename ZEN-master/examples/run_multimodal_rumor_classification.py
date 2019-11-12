import torch
import torch.nn as nn
from ZEN.modeling import ZenModel, ZenForSequenceClassification
from torchvision import models


# define models.
class FusionModel(nn.Module):
    """Fuse vgg and ZEN"""

    def __init__(self, config, num_labels=2):
        super(FusionModel, self).__init__()
        vgg = models.vgg19_bn()
        fc_features = vgg.classifier[-1].in_features
        vgg.classifier[-1] = nn.Linear(fc_features, num_labels)
        self.vgg = vgg
        self.zen = ZenForSequenceClassification.from_pretrained(args.bert_model, num_labels=num_labels, multift = args.multift)
        self.num_labels = num_labels
        self.linear = nn.Linear(2, 2)

    def forward(self, batched_imgs, input_ids, input_ngram_ids, ngram_position_matrix, token_type_ids=None,
                attention_mask=None, labels=None, head_mask=None):
        zen_logits = self.zen(input_ids, input_ngram_ids, ngram_position_matrix, token_type_ids=None,
                              attention_mask=None, labels=None, head_mask=None)
        vgg_logits = self.vgg(batched_imgs)
        all_logits = zen_logits + vgg_logits

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(all_logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return zen_logits, vgg_logits



