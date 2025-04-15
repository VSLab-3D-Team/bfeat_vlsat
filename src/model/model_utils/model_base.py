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
            'epoch': self.epoch,
            'eva_res': self.eva_res,
            'timestamp': time.strftime("%Y%m%d_%H%M%S")
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
            
            try:
                self.epoch = data.get('epoch', 0)
            except:
                self.epoch = 0
                
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
        loaded = True
        
        best_model_info_path = os.path.join(self.saving_pth, 'best_model_info.txt')
        
        config_files = [f for f in os.listdir(self.saving_pth) if f.startswith('config_epoch') and f.endswith('.pth')]
        
        if not config_files:
            print('\tNo saved models found')
            return False
        
        if best:
            if os.path.exists(best_model_info_path):
                with open(best_model_info_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_best_line = lines[-1]
                        suffix_info = last_best_line.strip().split('Suffix: ')[-1]
                        print(f'\tLoading best model with suffix: {suffix_info}')
                        config_path = self.config_path + suffix_info
                    else:
                        print('\tBest model info file exists but is empty, loading latest model instead')
                        best = False
            else:
                print('\tNo best model info found, loading latest model instead')
                best = False
        
        if not best:
            config_files.sort(key=lambda x: (
                int(x.split('_epoch')[1].split('_')[0]),  
                x.split('_epoch')[1].split('_')[1]       
            ), reverse=True)
            
            latest_config = config_files[0]
            suffix_info = latest_config.replace('config', '')
            print(f'\tLoading latest model: {latest_config}')
            config_path = self.config_path + suffix_info
        
        self.iteration, self.eva_res = self.loadConfig(config_path)
        print(f'\tLoaded model from iteration {self.iteration} with evaluation result {self.eva_res:.4f}')
        
        for name, model in self._modules.items():
            skip = False
            for k in self.skip_names:
                if name.find(k) != -1:
                    skip = True
            if skip is False:
                model_path = os.path.join(self.saving_pth, name + suffix_info)
                loaded &= self.loadWeights(model, model_path)
        
        optimizer_path = os.path.join(self.saving_pth, 'optimizer' + suffix_info)
        if os.path.exists(optimizer_path):
            data = torch.load(optimizer_path)
            self.optimizer.load_state_dict(data['optimizer'])
            print(f'\tResumed optimizer from {suffix_info}', flush=True)
        
        scheduler_path = os.path.join(self.saving_pth, 'lr_scheduler' + suffix_info)
        if os.path.exists(scheduler_path):
            data = torch.load(scheduler_path)
            self.lr_scheduler.load_state_dict(data['lr_scheduler'])
            print(f"\tResumed lr scheduler from {suffix_info}", flush=True)
        
        if loaded:
            print('\tModel loaded successfully!\n')
        else:
            print('\tModel loading failed!\n')
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