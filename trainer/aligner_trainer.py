#!/usr/bin/python
# -*- coding:utf-8 -*-
from utils import register as R
from .abs_trainer import Trainer


@R.register('AlignerTrainer')
class AlignerTrainer(Trainer):

    def __init__(self, model, train_loader, valid_loader, criterion: str, config: dict, save_config: dict):
        super().__init__(model, train_loader, valid_loader, config, save_config)
        self.criterion = criterion

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)
    
    def _train_epoch_end(self, device):
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'update_epoch'):
            dataset.update_epoch()
        return super()._train_epoch_end(device)

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        loss_dict = self.model(
            batch['X'], batch['S'], batch['A'], batch['block_lengths'], batch['lengths']
        )
        loss = loss_dict['S']# loss_dict['X'] + loss_dict['S'] + loss_dict['A']

        log_type = 'Validation' if val else 'Train'

        for key in loss_dict:
            self.log(f'{key}/{log_type}', loss_dict[key], batch_idx, val)

        if not val:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr, batch_idx, val)

        return loss