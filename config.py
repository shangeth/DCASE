import pprint
TRAINING_CONFIG = {
    'dataset_name' : 'timit_height',
    'task' : 'regression',
    'audio_features' : 'mel',
    'audio_sample_rate' : 16000,
    'timit_time_len', 200,
    'batch_size' : 128, 
    'augment' : True,
    'val_split_ratio' : 0.1,
    'data_dir' : '/home/shangeth/Downloads/dump/save_dir',
    'model_type' : 'cnn-lstm',
    'pretrained' : False,
    'lr' : 1e-3, 
    'epochs' : 100,
    'log_path' : 'logging/',
    'save_model_file' : 'trained_model.pt',
    'tensorboard_path' : 'logging/runs/'
}

if __name__ == "__main__":
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(TRAINING_CONFIG)