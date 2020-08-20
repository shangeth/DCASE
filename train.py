from dataset.Dataset20191b import Spectral_Dataset, RawWaveDataset
from dataset.TimitDataset import Timit_Dataset
from dataset.utils import get_dataloader
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from Trainer import dcase20191b_trainer, timit_trainer

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import logging

from config import TRAINING_CONFIG
from model.model_lookup import MODEL_LOOKUP
import torch.utils.data as data
import sys
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(filename=f'logging/logs/training_log_{datetime.now().strftime("%Y-%m-%d-%H:%M")}.log', 
                    level=logging.INFO,
                    format='%(asctime)s: %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S')
logger = logging.getLogger()

if __name__ == "__main__":

    dataset_name = TRAINING_CONFIG['dataset_name']
    task = TRAINING_CONFIG['task']
    DATA_DIR = TRAINING_CONFIG['data_dir']
    audio_features = TRAINING_CONFIG['audio_features'] 
    audio_sample_rate = TRAINING_CONFIG['audio_sample_rate'] 
    batch_size = TRAINING_CONFIG['batch_size']
    augment_bool = TRAINING_CONFIG['augment']
    val_split_ratio = TRAINING_CONFIG['val_split_ratio']
    model_type = TRAINING_CONFIG['model_type']
    pretrained = TRAINING_CONFIG['pretrained']
    lr = TRAINING_CONFIG['lr']
    epochs = TRAINING_CONFIG['epochs']
    log_path = TRAINING_CONFIG['log_path']
    save_model_file = TRAINING_CONFIG['save_model_file']
    tensorboard_path = TRAINING_CONFIG['tensorboard_path']

    if dataset_name == 'timit_height':
        train_set = Timit_Dataset(DATA_DIR+'/train')
        

        trainloader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
        valid_set = Timit_Dataset(DATA_DIR+'/valid', train=False)
        

        valloader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        test_set = Timit_Dataset(DATA_DIR+'/test', train=False)
        # plt.subplot(3, 1, 1)
        # plt.imshow(train_set[random.randint(0,100)][0].squeeze(0).numpy()[:, :])
        # plt.subplot(3, 1, 2)
        # plt.imshow(valid_set[random.randint(0,100)][0].squeeze(0).numpy()[:, :])
        # plt.subplot(3, 1, 3)
        # plt.imshow(test_set[random.randint(0,100)][0].squeeze(0).numpy()[:, :])
        # plt.show()
        testloader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        criterion = nn.MSELoss().to(device)
        class_num = 1
        fs = None
        ns = None
        # sys.exit(0)
    else:    
        if audio_features == 'raw':
            dataset = RawWaveDataset(DATA_DIR)
            spectral = False
        else:
            dataset = Spectral_Dataset(DATA_DIR)
            spectral = True
        dataset.print_stats()

        print(f'Data Shape = {dataset[0][0].shape}')

        fs = dataset.fs
        ns = dataset.ns
        trainloader, valloader = get_dataloader(dataset, train_bs=batch_size, valid_bs=batch_size, 
                                                val_ratio=val_split_ratio, augment=augment_bool, spectral=spectral,pretrained=pretrained)
        criterion = nn.CrossEntropyLoss()
        class_num = dataset.class_num
    print(f'Dataloader shape = {next(iter(trainloader))[0].shape}')

    model_class = MODEL_LOOKUP[task][audio_features][model_type]
    model = model_class(class_num, fs, ns).to(device)
    # model = torch.nn.DataParallel(model).to(device)
    # print(model)
    model.print_summary()

    # test_x = torch.randn(1, 1, fs*ns)
    # y_test = model(test_x.to(device))
    # print(y_test.shape)

    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # dcase20191b_trainer.train(model, trainloader, valloader, criterion, optimizer, 
    #                           epochs, logger, tensorboard_path, log_path, save_model_file)
    # dcase20191b_trainer.test(model, valloader, dataset, logger, log_path, save_model_file)
    timit_trainer.train(model, trainloader, valloader, criterion, optimizer, 
                              epochs, logger, tensorboard_path, log_path, save_model_file)
    # timit_trainer.test(model, testloader, logger, log_path, save_model_file)

    timit_trainer.inference(model, trainloader, 'Train', logger, log_path, save_model_file)
    timit_trainer.inference(model, valloader, 'Val', logger, log_path, save_model_file)
    timit_trainer.inference(model, testloader, 'Test', logger, log_path, save_model_file)


