import os
from tkinter import N
import torch
import torch.nn as nn
import collections
from pathlib import Path
import time

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()
        self.name = name
        self.config = config
        self.exp = config.exp
        self.epoch = -1
        self.iteration = 0
        self.eva_res = 0
        self.best_suffix = '_best.pth'
        self.suffix = '.pth'
        self.skip_names = ['loss']    
        self.saving_pth = os.path.join(config.PATH, 'ckp', name, self.exp)
        Path(self.saving_pth).mkdir(parents=True, exist_ok=True)
        self.config_path = os.path.join(self.saving_pth, 'config')
        
    def saveConfig(self, path):
        torch.save({
            'iteration': self.iteration,
            'eva_res' : self.eva_res
        }, path)
        
    def loadConfig(self, path):
        if os.path.exists(path):
            if torch.cuda.is_available():
                data = torch.load(path)
            else:
                data = torch.load(path, map_location=lambda storage, loc: storage)
                
            try:
                eva_res = data['eva_res']
            except:
                print('Target saving config file does not contain eva_res!')
                eva_res = 0
                
            return data['iteration'], eva_res
        else:
            return 0, 0
        
    def save(self):
        print('\nSaving %s...' % self.name)
        
        current_epoch = getattr(self, 'epoch', 0)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        unique_suffix = f"_epoch{current_epoch}_{timestamp}"
        
        performance_info = f"_IoU{self.eva_res:.4f}"
        save_suffix = unique_suffix + performance_info
        
        is_best = False
        if not os.path.exists(self.config_path+self.best_suffix):
            print('No previous best model found. This will be marked as the best.\n')
            is_best = True
        else:
            print('Found the previous best model.')
            _, prev_best_eva_res = self.loadConfig(self.config_path+self.best_suffix)
            print('current v.s. previous best: {:1.4f} v.s. {:1.4f}'.format(self.eva_res, prev_best_eva_res))
            if self.eva_res > prev_best_eva_res:
                print('Current IoU is better. This will be marked as the new best.\n')
                is_best = True
            else:
                print('Previous best IoU is still better.\n')
        
        self.saveConfig(self.config_path + save_suffix)
        
        for name, model in self._modules.items():
            skip = False
            for k in self.skip_names:
                if name.find(k) != -1:
                    skip = True
            if skip is False:
                self.saveWeights(model, os.path.join(self.saving_pth, name + save_suffix))
        
        torch.save({'optimizer': self.optimizer.state_dict()}, 
                os.path.join(self.saving_pth, 'optimizer' + save_suffix))
        torch.save({'lr_scheduler': self.lr_scheduler.state_dict()}, 
                os.path.join(self.saving_pth, 'lr_scheduler' + save_suffix))
        
        if is_best:
            best_marker_suffix = save_suffix + "_best"
            self.saveConfig(self.config_path + best_marker_suffix)
            
            with open(os.path.join(self.saving_pth, 'best_model_info.txt'), 'a') as f:
                f.write(f"Date: {timestamp}, Epoch: {current_epoch}, IoU: {self.eva_res:.4f}, Suffix: {save_suffix}\n")
                
    def load(self, best=False):
        print('\nLoading %s model...' % self.name)
        loaded=True
        
        if best:
            suffix = self.best_suffix
        else:
            if os.path.exists(self.config_path+self.best_suffix) and best:
                print('\tTrying to load the best model')
                suffix = self.best_suffix
            elif not os.path.exists(self.config_path+self.suffix) and os.path.exists(self.config_path+self.best_suffix):
                print('\tNo checkpoints, but has saved best model. Load the best model')
                suffix = self.best_suffix
            elif os.path.exists(self.config_path+self.suffix) and os.path.exists(self.config_path+self.best_suffix):
                print('\tFound checkpoint model and the best model. Comparing itertaion')
                iteration, _= self.loadConfig(self.config_path + self.suffix)
                iteration_best, _= self.loadConfig(self.config_path + self.best_suffix)
                if iteration > iteration_best:
                    print('\tcheckpoint has larger iteration value. Load checkpoint')
                    suffix = self.suffix
                else:
                    print('\tthe best model has larger iteration value. Load the best model')
                    suffix = self.best_suffix
            elif os.path.exists(self.config_path+self.suffix):
                print('\tLoad checkpoint')
                suffix = self.suffix
            else:
                print('\tNo saved model found')
                return False

        self.iteration, self.eva_res = self.loadConfig(self.config_path + suffix)
        for name,model in self._modules.items():
            skip = False
            for k in self.skip_names:
                if name.find(k) != -1:
                    skip = True
            if skip is False:
                #import ipdb; ipdb.set_trace()
                loaded &= self.loadWeights(model, os.path.join(self.saving_pth, name + suffix))
        
        if os.path.exists(os.path.join(self.saving_pth,'optimizer'+suffix)):
            data = torch.load(os.path.join(self.saving_pth,'optimizer'+suffix))
            self.optimizer.load_state_dict(data['optimizer'])
            print(f'resume optimizer from {suffix}', flush=True)
        
        if os.path.exists(os.path.join(self.saving_pth,'lr_scheduler'+suffix)):
            data = torch.load(os.path.join(self.saving_pth,'lr_scheduler'+suffix))
            self.lr_scheduler.load_state_dict(data['lr_scheduler'])
            print(f"resume lr scehduler from {suffix}", flush=True)
            
        if loaded:
            print('\tmodel loaded!\n')
        else:
            print('\tmodel loading failed!\n')
        return loaded
       
    def load_pretrain_model(self, path, skip_names=["predictor"], is_freeze=True):    
        loaded = True
        for name,model in self._modules.items():
            skip = False
            for k in skip_names:
                if name.find(k) != -1:
                    skip = True
            if skip is False:
                loaded &= self.loadWeights(model, os.path.join(path, name + '_best.pth'))
                if is_freeze:
                    for k, v in model.named_parameters():
                        v.requires_grad = False
        
        if loaded:
            print('\tmodel loaded!\n')
        else:
            print('\tmodel loading failed!\n')

    
    def saveWeights(self, model, path):
        if isinstance(model, nn.DataParallel):
            torch.save({
                'model': model.module.state_dict()
            }, path)
        else:
            torch.save({
                'model': model.state_dict()
            }, path)
    
    def loadWeights(self, model, path):
        # print('isinstance(model, nn.DataParallel): ',isinstance(model, nn.DataParallel))
        if os.path.exists(path):
            if torch.cuda.is_available():
                data = torch.load(path)
            else:
                data = torch.load(path, map_location=lambda storage, loc: storage)
                
            
            new_dict = collections.OrderedDict()
            if isinstance(model, nn.DataParallel):
                for k,v in data['model'].items():                    
                    if k[:6] != 'module':
                        name = 'module.' + k
                        new_dict [name] = v
                model.load_state_dict(new_dict)
            else:
                for k,v in data['model'].items():                    
                    if k[:6] == 'module':
                        name = k[7:]
                        new_dict [name] = v
                model.load_state_dict(data['model'])
            return True
        else:
            return False