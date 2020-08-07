from dataset.Dataset20191b import RawWaveDataset
from dataset.utils import get_dataloader
from model.raw import CNN1d_1s
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DATA_DIR = 'data/audio'
dataset = RawWaveDataset(DATA_DIR)
dataset.print_stats()
fs = dataset.fs
ns = dataset.ns
trainloader, valloader = get_dataloader(dataset)


model = CNN1d_1s(dataset.class_num, fs, ns)
model.to(device)
model.print_summary()

test_x = torch.randn(1, 1, fs*ns)
y_test = model(test_x.to(device))
print(y_test.shape)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)



#---------------------------------

writer = SummaryWriter()
# no of epochs
epochs = 20

# loss lists to visualize
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

iter = 0
# training loop
for epoch in range(epochs):
    t1 = time.time()
    epoch_loss = 0
    epoch_accuracy = 0
    model.train() # training mode
    for batch in tqdm(trainloader):
        model.train()
        batch_x, batch_y1, batch_y2 = batch
        batch_x, batch_y1, batch_y2 = batch_x.to(device), batch_y1.to(device), batch_y2.to(device)
        y_hat1 = model(batch_x) # forward pass throough model
        loss = criterion(y_hat1, batch_y1) 
        # + criterion(y_hat2, batch_y2) # compute the loss
        epoch_loss += loss.item()
        optimizer.zero_grad() # zero the previous computed gradients
        loss.backward() # calculate loss gradients wrt all parameters
        optimizer.step() # update all parameters in model with optimizer

        # calculate the accuracy of prediction
        max_vals, max_indices = torch.max(y_hat1,1)
        train_batch_accuracy = (max_indices == batch_y1).sum().data.cpu().numpy()/max_indices.size()[0]
        epoch_accuracy += train_batch_accuracy
        writer.add_scalar('Loss/train', loss.item(), iter)
        writer.add_scalar('Accuracy/train', train_batch_accuracy, iter)
        # print(loss.item(), train_batch_accuracy)
        iter += 1
    epoch_loss = epoch_loss/len(trainloader)
    epoch_accuracy = epoch_accuracy/len(trainloader)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    model.eval() # evaluation mode
    with torch.no_grad(): # no gradients/back propagation for evaluation
        val_loss = 0
        val_accuracy = 0
        for batch in valloader:
            batch_x, batch_y1, batch_y2 = batch
            batch_x, batch_y1, batch_y2 = batch_x.to(device), batch_y1.to(device), batch_y2.to(device)
            y_hat1 = model(batch_x) # forward pass throough model
            loss = criterion(y_hat1, batch_y1) 
            # + criterion(y_hat2, batch_y2)
            val_loss += loss.item()
            
            # validation accuracy
            max_vals, max_indices = torch.max(y_hat1,1)
            val_batch_accuracy = (max_indices == batch_y1).sum().data.cpu().numpy()/max_indices.size()[0]
            val_accuracy += val_batch_accuracy

        val_loss = val_loss/len(valloader)
        val_accuracy = val_accuracy/len(valloader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
    t2 = time.time()
    t = t2-t1
    model.save_state_dict('trained_model.pt')
    print(f'\rEpoch : {epoch+1:02}\tLoss : {epoch_loss:.4f}\tAccuracy : {epoch_accuracy:.4f}\tVal Loss : {val_loss:.4f}\tVal Accuracy : {val_accuracy:.4f}\tTime : {t:.2f} s')

writer.close()