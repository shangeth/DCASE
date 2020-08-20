from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from tqdm import tqdm 
import time
import torch
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import gc
from pytorch_lightning.metrics.regression import MAE, MSE, RMSE

def train(model, trainloader, valloader, criterion, optimizer, epochs, logger, tensorboard_path, log_path, save_model_file):
    writer = SummaryWriter(f'{tensorboard_path}{datetime.now().strftime("%m-%d-%Y_%H:%M:%S")}')

    criterion2 = RMSE().to(device)
    criterion3 = MAE().to(device)

    val_loss_min = np.inf
    scheduler_step = 0
    # loss lists to visualize
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-2, cycle_momentum=False)
    # training loop
    m, M = (144.78, 203.2)
    for epoch in range(epochs):
        t1 = time.time()
        epoch_loss = 0
        epoch_accuracy = 0
        model.train() # training mode
        for batch in tqdm(trainloader):
            model.train()
            batch_x, batch_y = batch
            batch_x, batch_y = batch_x.float().to(device), batch_y.view(-1).float().to(device)
            y_hat = model(batch_x) # forward pass throough model
            # loss = criterion(y_hat.view(-1), batch_y) + criterion2(y_hat.view(-1), batch_y) + 
            loss = criterion2(y_hat.view(-1), batch_y)
            # + criterion(y_hat2, batch_y2) # compute the loss
            epoch_loss += loss.item()
            optimizer.zero_grad() # zero the previous computed gradients
            loss.backward() # calculate loss gradients wrt all parameters
            optimizer.step() # update all parameters in model with optimizer
            # scheduler.step()
            # writer.add_scalar('LR/lr', float(scheduler.get_lr()[0]), scheduler_step)
            scheduler_step += 1
            # torch.cuda.empty_cache()
            

            # calculate the accuracy of prediction
            mae = MAE()(y_hat.view(-1)* (M-m) + m, batch_y* (M-m) + m).item()
            epoch_accuracy += mae
            
            # print(loss.item(), train_batch_accuracy)
            
        epoch_loss = epoch_loss/len(trainloader)
        epoch_accuracy = epoch_accuracy/len(trainloader)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        writer.add_scalar('MSE/train', epoch_loss, epoch)
        writer.add_scalar('MAE/train', epoch_accuracy, epoch)
        # torch.cuda.empty_cache()
        # gc.collect()

        model.eval() # evaluation mode
        with torch.no_grad(): # no gradients/back propagation for evaluation
            val_loss = 0
            val_accuracy = 0
            for batch in valloader:
                batch_x, batch_y = batch
                batch_x, batch_y = batch_x.float().to(device), batch_y.float().view(-1).to(device)
                y_hat = model(batch_x)
                # loss = criterion(y_hat.view(-1), batch_y) + criterion2(y_hat.view(-1), batch_y) + 
                loss = criterion2(y_hat.view(-1), batch_y)
                # + criterion(y_hat2, batch_y2)
                val_loss += loss.item()
                
                # validation accuracy
                mae = MAE()(y_hat.view(-1)* (M-m) + m, batch_y* (M-m) + m).item()
                val_accuracy += mae
                
            val_loss = val_loss/len(valloader)
            val_accuracy = val_accuracy/len(valloader)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            writer.add_scalar('MSE/val', val_loss, epoch)
            writer.add_scalar('MAE/val', val_accuracy, epoch)

            if val_loss < val_loss_min:
                torch.save(model.state_dict(), log_path+save_model_file)
                logger.info(f'Validation loss reduced from {val_loss_min} to  {val_loss}, saving model...')
                tqdm.write(f'Validation loss reduced from {val_loss_min} to  {val_loss}, saving model...')
                val_loss_min = val_loss
        t2 = time.time()
        t = t2-t1
        logger.info(f'Epoch : {epoch+1:02}\tLoss : {epoch_loss:.4f}\tMAE : {epoch_accuracy:.4f}\tVal Loss : {val_loss:.4f}\tVal MAE : {val_accuracy:.4f}\tTime : {t:.2f} s')
        tqdm.write(f'\nEpoch : {epoch+1:02}\tLoss : {epoch_loss:.4f}\tMAE : {epoch_accuracy:.4f}\tVal Loss : {val_loss:.4f}\tVal MAE : {val_accuracy:.4f}\tTime : {t:.2f} s\n')

    writer.close()

    plt.figure(figsize=(20,10))
    plt.title('Training Learning Curve')

    plt.subplot(1,2,1)
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy loss')
    plt.plot(train_losses, label='Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(train_accuracies, label='Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.legend()
    plt.savefig(log_path + 'training_curve.png')
    plt.show()

def test(model, loader, logger, log_path, save_model_file):
    model.load_state_dict(torch.load(log_path+save_model_file))
    true_labels = []
    predictions = []
    model.eval() # evaluation mode
    with torch.no_grad(): # no gradients/back propagation for evaluation
        for batch in loader:
            batch_x, batch_y = batch
            batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
            y_hat = model(batch_x) # forward pass throough model
            
            predictions += y_hat.cpu().detach().view(-1)
            true_labels += batch_y.cpu().detach().view(-1)

    prediction = torch.Tensor(predictions)
    label = torch.Tensor(true_labels)

    mse = MSE()(prediction, label)
    mae = MAE()(prediction, label)
    rmse = RMSE()(prediction, label)

    print(f'\n\nTest Dataset :\tMSE : {mse}\tMAE : {mae}\tRMSE: {rmse}')
    logger.info(f'\n\nTest Dataset :\tMSE : {mse}\tMAE : {mae}\tRMSE: {rmse}')


def inference(model, loader, label_name, logger, log_path, save_model_file):
    m, M = (144.78, 203.2)
    model.load_state_dict(torch.load(log_path+save_model_file))
    true_labels = []
    predictions = []
    model.eval() # evaluation mode
    with torch.no_grad(): # no gradients/back propagation for evaluation
        for batch in loader:
            batch_x, batch_y = batch
            batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
            y_hat = model(batch_x) # forward pass throough model
            
            predictions += y_hat.cpu().detach().view(-1)
            true_labels += batch_y.cpu().detach().view(-1)

    prediction = torch.Tensor(predictions) * (M-m) + m
    label = torch.Tensor(true_labels) * (M-m) + m

    mse = MSE()(prediction, label)
    mae = MAE()(prediction, label)
    rmse = RMSE()(prediction, label)

    print(f'\n{label_name} Dataset :\tMSE : {mse}\tMAE : {mae}\tRMSE: {rmse}')
    logger.info(f'\n{label_name} Dataset :\tMSE : {mse}\tMAE : {mae}\tRMSE: {rmse}')

