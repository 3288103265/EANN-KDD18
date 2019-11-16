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

    def __init__(self, bert_model, num_labels=10):
        super(FusionModel, self).__init__()
        vgg = models.vgg19_bn()
        fc_features = vgg.classifier[-1].in_features
        vgg.classifier[-1] = nn.Linear(fc_features, num_labels)
        self.vgg = vgg
        self.zen = ZenForSequenceClassification.from_pretrained(bert_model, num_labels=num_labels)
        self.num_labels = num_labels

    def forward(self, batched_imgs, input_ids, input_ngram_ids, ngram_position_matrix, token_type_ids=None,
                attention_mask=None, labels=None, head_mask=None):
        zen_logits = self.zen(input_ids, input_ngram_ids, ngram_position_matrix, token_type_ids=None,
                              attention_mask=None, labels=None, head_mask=None)
        # 不给 label，输出logits。
        vgg_logits = self.vgg(batched_imgs)
        all_logits = zen_logits + vgg_logits

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(all_logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return all_logits


# Prepare data
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

F = open('all_dict.pkl', 'rb')
all_dict = pickle.load(F)
F.close()
all_sets = FusionFolder("/home/wangpenghui/Datasets/WeiboRumorSet/", all_dict, transform=data_transforms)

# split dataset into trainset and test set.
train_size = int(0.6 * len(all_sets))
test_size = len(all_sets) - train_size
torch.manual_seed(1000)
train_dataset, test_dataset = torch.utils.data.random_split(all_sets, [train_size, test_size])
F = open('train_set.pkl', 'wb')
pickle.dump(train_dataset, F)
F.close()
train_loader = DataLoader(train_dataset, batch_size=12, num_workers=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=12, num_workers=4, shuffle=True)


#set parameters.
num_epoches = 30
bert_model = '/home/wangpenghui/EANN-KDD18/ZEN-master/models/ZEN_ft_DC_v0.1.0'

fusion_model = FusionModel(bert_model, num_labels=10).cuda()

optimizer = optim.SGD([{'params': fusion_model.vgg.parameters()},
                           {'params': fusion_model.zen.classifier.parameters(), 'lr': 7e-4}],
                          lr=0.001, momentum=0.9)

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

