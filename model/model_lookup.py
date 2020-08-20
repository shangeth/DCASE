from model.raw import *
from model.spectral import *
from model.regression import *
import pprint

MODEL_LOOKUP = {
    'regression':{
        'mel' : {
            '1d' : CNN_Regression_1D,
            '2d' : CNN_Regression_2D,
            'lstm' : LSTM_Regression
        }
    },
    'classification' : {
        'raw' : {
            '1d' : CNN1d_1s, # 1 second
            '2d' : CNN1D # 1d complete
                },

        'mel' : {
            '1d' : CNN_MEL_1D,
            '2d' : CNN_MEL_2D,
            'vgg': VGG,
            'squeezenet' : SqueezeNet
                },
        'mfcc' : {
            '1d' : CNN_MFCC_2D,
            '2d' : CNN_MFCC_2D,
            'vgg': VGG,
            'squeezenet' : SqueezeNet
                }
    }
}


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(depth=4)
    pp.pprint(MODEL_LOOKUP)