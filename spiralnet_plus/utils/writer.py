import os
import time
import torch
import json
from glob import glob


class Writer:
    def __init__(self, args=None):
        self.args = args

        if self.args is not None:
            tmp_log_list = glob(os.path.join(args.out_dir, 'log*'))
            if len(tmp_log_list) == 0:
                self.log_file = os.path.join(
                    args.out_dir, 'log_{:s}.txt'.format(
                        time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))
            else:
                self.log_file = tmp_log_list[0]

    def print_info(self, info):
        message = 'Epoch: {}/{}, Duration: {:.3f}s\n'.format(info['current_epoch'], info['epochs'], info['t_duration'])

        for (loss_name, loss_value) in info['train_loss'].items():
            message += loss_name + ': {:.4f}, '.format(loss_value)
        for (loss_name, loss_value) in info['test_loss'].items():
            message += loss_name + ': {:.4f}, '.format(loss_value)
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def save_checkpoint(self, model, optimizers, schedulers, epoch):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': [optimizer.state_dict() for optimizer in optimizers],
                'scheduler_state_dict': [scheduler.state_dict() for scheduler in schedulers],
            },
            os.path.join(self.args.checkpoints_dir,
                         'checkpoint_{:03d}.pt'.format(epoch)))
