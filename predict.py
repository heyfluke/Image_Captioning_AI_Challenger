from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os, sys
from six.moves import cPickle

import opts
import models
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

class Predictor:
    def __init__(self, opt, infos_path):
        self.opt = opt
        # Load infos
        with open(infos_path) as f:
            self.infos = cPickle.load(f)

        ignore = ["batch_size", "start_from"]

        for k in vars(self.infos['opt']).keys():
            if k not in ignore:
                if k in vars(opt):
                    assert vars(opt)[k] == vars(self.infos['opt'])[k], k + ' option not consistent'
                else:
                    vars(opt).update({k: vars(self.infos['opt'])[k]}) # copy over options from model

    def predict_with_image_folder(self, image_folder):
        # Setup the model
        print("self.opt", self.opt)
        model = models.setup(self.opt)
        if self.opt.use_cpu:
            state_dict = torch.load(self.opt.model, map_location={'cuda:0':'cpu'})
            print('state_dict keys', state_dict.keys())
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(torch.load(self.opt.model))
        if not self.opt.use_cpu:
            model.cuda()
        model.eval()
        crit = utils.LanguageModelCriterion()

        # Create the Data Loader instance
        loader = DataLoaderRaw({'folder_path': image_folder, 
                                    'batch_size': 1,
                                    'cnn_model': self.opt.cnn_model,
                                    'use_cpu': self.opt.use_cpu,
                                    })
        # When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
        # So make sure to use the vocab in infos file.
        loader.ix_to_word = self.infos['vocab']

        print('vars opt', vars(self.opt))
        # Set sample options
        predictions = eval_utils.predict(model, crit, loader, vars(self.opt))
        print('len(predictions)', len(predictions))
        for idx,p in enumerate(predictions):
            print('#%d\n' % idx)
            for k in p.keys():
                print('\t%s: %s' % (k, p[k]))

def usage():
    sys.stderr.write('please see -h\n')
    sys.exit(1)

def main():
    # Input arguments and options
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--model', type=str, default='',
                    help='path to model to evaluate')
    parser.add_argument('--cnn_model', type=str,  default='resnet101',
                    help='resnet101, resnet152')
    parser.add_argument('--infos_path', type=str, default='',
                    help='path to infos to evaluate')
    parser.add_argument('--use_cpu', type=bool, default=False,
                    help='use cpu not not.')

    # Sampling options
    # For evaluation on a folder of images:
    parser.add_argument('--image_folder', type=str, default='', 
                    help='image path to predict.')

    opt = parser.parse_args()

    if not opt.image_folder:
        usage()

    p = Predictor(opt, opt.infos_path)
    p.predict_with_image_folder(opt.image_folder)

if __name__ == '__main__':
    main()

