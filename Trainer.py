import torch
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from Datasets.SMTGraghDataset import SMTGraghDataset
from Datasets.BaseDataset import BaseDataset
from Models.Simple.SimpleGNN import SimpleGNN
from Models.BaseModel import BaseModel
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import inspect
from Training.ProgressManager import ProgressManager
from Training.TensorBoardManager import TensorBoardManager 
from Training.MovingAvg import MovingAvg 
from Training.LRScheduler import LRScheduler
from Datasets.MultiThreadLoader import MultiThreadLoader
from Datasets.DatasetUtils import *
import os
from sklearn.model_selection import KFold

def batch_to_dict(batch):
    """将 PyTorch Geometric 的 Batch 对象转换为字典"""
    return {
        'x': batch.x,
        'edge_index': batch.edge_index,
        'batch': batch.batch,
        'y': batch.y
    }

class Trainer:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, 
                 model: BaseModel, 
                 train_set: BaseDataset,
                 eval_set: BaseDataset,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler | LRScheduler,
                 log_dir: str,
                 save_dir: str,
                 n_epochs: int,
                 moving_avg: int, 
                 eval_interval: int,
                ) -> None:
       
        self.train_set = train_set
        self.eval_set = eval_set
        self.model = model
        
        # 获取实际的模型对象来访问属性
        self.actual_model = model.module if hasattr(model, 'module') else model

        if not hasattr(lr_scheduler, 'update'):
            num_args = len(inspect.signature(lr_scheduler.step).parameters)
            if num_args == 1:
                lr_scheduler.update = lambda metric: lr_scheduler.step()
            else:
                lr_scheduler.update = lambda metric : lr_scheduler.step(metric)

        self.lr_scheduler = lr_scheduler
        self.log_dir =  log_dir
        self.save_dir = save_dir
        self.n_epochs = n_epochs
        self.moving_avg = moving_avg
        self.eval_interval = eval_interval

    def start(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pm_log_tags = self.actual_model.train_loss_names + ["LR"]
        tm_log_tags = self.actual_model.train_loss_names + self.actual_model.eval_loss_names + ["LR"]

        pm = ProgressManager(self.train_set.n_batches, self.n_epochs, 5, 2, custom_fields=pm_log_tags)
        tm = TensorBoardManager(self.log_dir, tags=tm_log_tags, value_types=["scalar"] * len(tm_log_tags))
        ma_losses = {name: MovingAvg(self.moving_avg) for name in self.actual_model.train_loss_names}

        best_loss = float('inf')

        for epoch in range(self.n_epochs):
            if epoch % 5 == 0:
                torch.cuda.empty_cache()

            loader = MultiThreadLoader(self.train_set, 3)
            for i, data_dict in enumerate(loader):
                data_dict = {k: v.to(self.device) if hasattr(v, 'to') else v 
                    for k, v in data_dict.items()}
                
                # 使用 actual_model 调用 trainStep
                loss_dict, output_dict = self.actual_model.trainStep(data_dict)

                for loss_name in self.actual_model.train_loss_names:
                    ma_losses[loss_name].update(loss_dict[loss_name])
                    loss_dict[loss_name] = ma_losses[loss_name].get()

                current_lr = self.actual_model.lr
                pm.update(epoch, i, LR=current_lr, **loss_dict)

            current_lr = self.actual_model.lr
            tm.log(pm.overall_progress, LR=current_lr, **loss_dict)

            self.lr_scheduler.update(loss_dict["Train_MSE"])

            if epoch % self.eval_interval == 0:
                eval_losses = self.evaluate(self.eval_set)
                tm.log(pm.overall_progress, **eval_losses)

                eval_loss = eval_losses["Eval_MSE"]
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    self.actual_model.saveTo(os.path.join(self.save_dir, "best.pth"))

        pm.close()

    def evaluate(self, dataset: BaseDataset, compute_avg: bool=True):
        n_batches = dataset.n_batches
        eval_losses = {name: torch.zeros(n_batches).to(DEVICE) for name in self.actual_model.eval_loss_names}
        self.model.eval()

        for i, data_dict in enumerate(dataset):
            # 使用 actual_model 调用 evalStep
            loss_dict, output_dict = self.actual_model.evalStep(data_dict)

            for name in self.actual_model.eval_loss_names:
                eval_losses[name][i] = loss_dict[name]

        self.model.train()

        if compute_avg:
            return {name: torch.mean(eval_losses[name]).item() for name in self.actual_model.eval_loss_names}

        return {name: eval_losses[name].cpu().numpy() for name in self.actual_model.eval_loss_names}
