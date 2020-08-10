from dataset.Dataset20191b import Spectral_Dataset, RawWaveDataset
from dataset.utils import get_dataloader
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import logging
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from config import TRAINING_CONFIG
from model.model_lookup import MODEL_LOOKUP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(filename=f'logging/logs/training_log_{datetime.now().strftime("%Y-%m-%d-%H:%M")}.log', 
                    level=logging.INFO,
                    format='%(asctime)s: %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S')

if __name__ == "__main__":
    
    DATA_DIR = TRAINING_CONFIG['data_dir']
    audio_features = TRAINING_CONFIG['audio_features'] 
    audio_sample_rate = TRAINING_CONFIG['audio_sample_rate'] 
    batch_size = TRAINING_CONFIG['batch_size']
    augment_bool = TRAINING_CONFIG['augment']
    val_split_ratio = TRAINING_CONFIG['val_split_ratio']
    model_type = TRAINING_CONFIG['model_type']
    lr = TRAINING_CONFIG['lr']
    epochs = TRAINING_CONFIG['epochs']
    log_path = TRAINING_CONFIG['log_path']
    save_model_file = TRAINING_CONFIG['save_model_file']
    tensorboard_path = TRAINING_CONFIG['tensorboard_path']


    if audio_features == 'raw':
        dataset = RawWaveDataset(DATA_DIR)
        spectral = False
    else:
        dataset = Spectral_Dataset(DATA_DIR)
        spectral = True
    dataset.print_stats()

    fs = dataset.fs
    ns = dataset.ns
    trainloader, valloader = get_dataloader(dataset, train_bs=batch_size, valid_bs=batch_size, 
                                            val_ratio=val_split_ratio, augment=augment_bool, spectral=spectral)

    model_class = MODEL_LOOKUP[audio_features][model_type]
    model = model_class(dataset.class_num, fs, ns)
    model.to(device)
    model.print_summary()

    # test_x = torch.randn(1, 1, fs*ns)
    # y_test = model(test_x.to(device))
    # print(y_test.shape)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)



    #---------------------------------

    writer = SummaryWriter(f'{tensorboard_path}{datetime.now().strftime("%m-%d-%Y_%H:%M:%S")}')

    val_loss_min = np.inf

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
            writer.add_scalar('Loss/step_train', loss.item(), iter)
            writer.add_scalar('Accuracy/step_train', train_batch_accuracy, iter)
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
            writer.add_scalar('Loss/epoch_val', val_loss, epoch)
            writer.add_scalar('Accuracy/epoch_val', val_accuracy, epoch)
            if val_loss < val_loss_min:
                torch.save(model.state_dict(), log_path+save_model_file)
                logging.info(f'Validation loss reduced from {val_loss_min} to  {val_loss}, saving model...')
                tqdm.write(f'Validation loss reduced from {val_loss_min} to  {val_loss}, saving model...')
                val_loss_min = val_loss
        t2 = time.time()
        t = t2-t1
        logging.info(f'Epoch : {epoch+1:02}\tLoss : {epoch_loss:.4f}\tAccuracy : {epoch_accuracy:.4f}\tVal Loss : {val_loss:.4f}\tVal Accuracy : {val_accuracy:.4f}\tTime : {t:.2f} s')
        tqdm.write(f'\nEpoch : {epoch+1:02}\tLoss : {epoch_loss:.4f}\tAccuracy : {epoch_accuracy:.4f}\tVal Loss : {val_loss:.4f}\tVal Accuracy : {val_accuracy:.4f}\tTime : {t:.2f} s')

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

    # ---------------------------------
    model.load_state_dict(torch.load(log_path+save_model_file))
    true_labels = []
    predictions = []
    model.eval() # evaluation mode
    with torch.no_grad(): # no gradients/back propagation for evaluation
        val_loss = 0
        val_accuracy = 0
        for batch in valloader:
            batch_x, batch_y1, batch_y2 = batch
            batch_x, batch_y1, batch_y2 = batch_x.to(device), batch_y1.to(device), batch_y2.to(device)
            y_hat1 = model(batch_x) # forward pass throough model
            _, max_indices = torch.max(y_hat1,1)
            predictions += max_indices.cpu().detach().numpy().reshape(-1).tolist()
            true_labels += batch_y1.cpu().detach().numpy().reshape(-1).tolist()

    acc = accuracy_score(true_labels, predictions)
    print(f'\nValidation Accuracy = {acc}\n')
    logging.info(f'\nValidation Accuracy = {acc}\n')

    clf_report = classification_report(true_labels, predictions, target_names=dataset.labels_list)
    print(f'\nValidation Classification Report :\n{clf_report}\n')
    logging.info(f'\nnValidation Classification Report\n{clf_report}\n')    

    print(f'\nValidation Confusion Matrix :\n')
    cf_matrix = confusion_matrix(true_labels, predictions)
    hm = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
                fmt='.2%', cmap='Blues', xticklabels=dataset.labels_list,
                yticklabels=dataset.labels_list)
    plt.show()
    fig = hm.get_figure()
    fig.savefig(log_path + 'confusion_matrix.png') 