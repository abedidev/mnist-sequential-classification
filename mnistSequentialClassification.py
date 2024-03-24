import numpy as np
import torch.optim as optim
import torchvision
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms
from tqdm import tqdm
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.input_size = 1
        self.hidden_size = 64
        self.num_layers = 2
        self.num_classes = 10

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.num_classes)

        # Attention mechanism components
        self.attention_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.context_vector = nn.Parameter(torch.randn(self.hidden_size, 1))

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.linear(torch.sum(out, dim=1))
        return out


torch.manual_seed(0)
num_epochs = 100
batch_size_train = 64
batch_size_test = 64

lr = .001
# lr = .0001

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
print("Device being used:", device, flush=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: x.view(-1, 1)),
])

train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train)

test_dataset = datasets.MNIST('../data', train=False, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test)

train_size = len(train_dataloader)

model = Model().to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=20, gamma=.5)

criterion = nn.CrossEntropyLoss().to(device)

for epoch in range(num_epochs):

    running_loss = 0.0

    y_trues = np.empty([0])
    y_preds = np.empty([0])

    model.train()

    for inputs, labels in tqdm(train_dataloader, disable=True):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        y_trues = np.append(y_trues, labels.detach().cpu().numpy())

        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        y_preds = np.append(y_preds, preds.detach().cpu().numpy())

    epoch_loss = running_loss / train_size

    print("[{}] Epoch: {}/{} Loss: {}".format('train', epoch + 1, num_epochs, epoch_loss), flush=True)
    print('\n' + str(confusion_matrix(y_trues, y_preds)), flush=True)
    print('\n' + str(accuracy_score(y_trues, y_preds)), flush=True)

    scheduler.step()

# TEST
running_loss = 0.0

y_trues = np.empty([0])
y_preds = np.empty([0])

model.eval()

for inputs, labels in tqdm(test_dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)

    loss = criterion(outputs, labels)

    running_loss += loss.item() * inputs.size(0)

    y_trues = np.append(y_trues, labels.detach().cpu().numpy())

    probs = nn.Softmax(dim=1)(outputs)
    preds = torch.max(probs, 1)[1]
    y_preds = np.append(y_preds, preds.detach().cpu().numpy())

epoch_loss = running_loss / train_size

print("[{}] Epoch: {}/{} Loss: {}".format('test', epoch + 1, num_epochs, epoch_loss), flush=True)
print('\n' + str(confusion_matrix(y_trues, y_preds)), flush=True)
print('\n' + str(accuracy_score(y_trues, y_preds)), flush=True)
