import torchvision.models as models
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch import optim
import pretrainedmodels

# Pretrained model
# modify last layer
model = models.vgg19_bn()
fc_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(fc_features, 2)

data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Randomly split dataset into train and test parts.
batch_size = 16
dataset = ImageFolder(root='/home/wangpenghui/Datasets/WeiboRumorSet', transform=data_transforms)
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

# Prepare for train.
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fc = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, 200, 0.1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training
epoch_nums = 50
# best_model_wts = model.state_dict()
best_acc = 0

for epoch in range(epoch_nums):
    running_loss = 0.0
    epoch_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):

        inputs = inputs.to(device)
        labels = labels.to(device)

        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fc(outputs, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    if epoch % 10 == 0:
        correct = 0
        total = 0
        for image_test, labels_test in test_loader:
            model.eval()
            image_test = image_test.to(device)
            labels_test = labels_test.to(device)
            outputs_test = model(image_test)
            _, prediction = torch.max(outputs_test, 1)
            correct += ((prediction == labels_test).sum()).item()
            total += labels_test.size(0)

        accuracy = correct / total
        print('epoch={}, running loss={:.5f}, accuracy={:.5f}'.format(epoch + 1, running_loss / 10,
                                                                      accuracy))

        if accuracy > best_acc:
            best_acc = accuracy
            # torch.save(model, '../outputs/model_epoch{}.pkl'.format(epoch))

print('Training finished')
