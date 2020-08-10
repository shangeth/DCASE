import pprint
TRAINING_CONFIG = {
    'audio_features' : 'mfcc',
    'audio_sample_rate' : 16000,
    'batch_size' : 128, 
    'augment' : True,
    'val_split_ratio' : 0.1,
    'data_dir' : 'data/mfcc',
    'model_type' : '1d',
    'pretrained' : False,
    'lr' : 1e-3, 
    'epochs' : 10,
    'log_path' : 'logging/',
    'save_model_file' : 'trained_model.pt',
    'tensorboard_path' : 'logging/runs/'
}

if __name__ == "__main__":
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(TRAINING_CONFIG)