import json
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ZEN.modeling import ZenModel, ZenForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from torchtext.data import Dataset
from torchvision import models
from torchvision.transforms import transforms
from tqdm import tqdm

from utils_multi_dataset import FusionFolder

# define models.
from utils_sequence_level_task import simple_accuracy, compute_metrics


class FusionModel(nn.Module):
    """Fuse vgg and ZEN"""

    def __init__(self, bert_model, num_labels=2):
        super(FusionModel, self).__init__()
        vgg = models.vgg19_bn()
        fc_features = vgg.classifier[-1].in_features
        vgg.classifier[-1] = nn.Linear(fc_features, 32)
        self.vgg = vgg
        zen = ZenForSequenceClassification.from_pretrained(bert_model, num_labels=num_labels)
        for param in zen.parameters():
            param.requires_grad = False
        zen.classifier = nn.Linear(768, 32)
        self.zen = zen
        self.num_labels = num_labels
        self.linear = nn.Linear(64, 2)

    def forward(self, batched_imgs, input_ids, input_ngram_ids, ngram_position_matrix, token_type_ids=None,
                attention_mask=None, labels=None, head_mask=None):

        zen_out = self.zen(input_ids, input_ngram_ids, ngram_position_matrix, token_type_ids=None,
                              attention_mask=None, labels=None, head_mask=None)
        # 不给 label，输出logits。
        vgg_out = self.vgg(batched_imgs)
        all_out = torch.cat((zen_out, vgg_out), dim=1)
        all_logits = self.linear(all_out)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(all_logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return all_logits


# Prepare data

F_train = open('train_set.pkl', 'rb')
train_dataset = pickle.load(F_train)
F_train.close()

F_test = open('test_set.pkl', 'rb')
test_dataset = pickle.load(F_test)
F_test.close()

train_loader = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, num_workers=4, shuffle=True)


#set parameters.
num_epoches = 10
bert_model = '/home/wangpenghui/EANN-KDD18/ZEN-master/models/output/checkpoint-6750'

fusion_model = FusionModel(bert_model, num_labels=2).cuda()

optimizer = optim.SGD(filter(lambda p: p.requires_grad, fusion_model.parameters()), lr=0.0001, momentum=0.9)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_gpu = torch.cuda.device_count()
fusion_model = torch.nn.DataParallel(fusion_model)

# Train !!
for epoch in range(num_epoches):
    # vgg fineturn.
    running_loss = 0.0
    epoch_loss = 0.0

    for step, batch in enumerate(tqdm(train_loader, desc="Iteration", disable=False)):
        batch = tuple(t.to(device) for t in batch)

        optimizer.zero_grad()
        loss = fusion_model(batch[0], batch[2], batch[3], batch[4], labels=batch[1])
        if n_gpu > 1:
            loss = loss.mean()

        running_loss += loss.item()
        loss.backward()
        optimizer.step()

    if epoch % 2 == 0:
        preds = []
        out_label_ids = None

        for step, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            batch = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                logits = fusion_model(batch[0], batch[2], batch[3], batch[4])

                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                    out_label_ids = batch[1].detach().cpu().numpy()
                else:
                    preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)
                    out_label_ids = np.append(out_label_ids, batch[1].detach().cpu().numpy(), axis=0)

        preds = np.argmax(preds[0], axis=1)
        print(compute_metrics('abc', preds, out_label_ids))

