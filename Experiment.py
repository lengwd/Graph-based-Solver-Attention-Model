from Trainer import Trainer
from DynamicConfig import DynamicConfig
from Models.Simple.SimpleGNN import SimpleGNN
import torch
from Datasets.SMTGraghDataset import SMTGraghDataset
from Training.LRScheduler import LRScheduler
from Datasets.DatasetUtils import *
from datetime import datetime
import os
import pandas as pd
from evaluate_model import evaluate_model


class Experiment:
    def __init__(self, comments: str, train_data_dir: str, eval_data_dir: str, test_data_dir: str, is_test: bool = False):
        self.comments = comments
        self.is_test = is_test
        self.eval_data_dir = eval_data_dir
        self.test_data_dir = test_data_dir

        self.model_cfg = DynamicConfig(SimpleGNN,
                                       node_features_dim = 20,
                                       num_solvers=6,
                                       optimizer_cls=torch.optim.Adam,
                                       optimizer_args={"lr": 1e-5},
                                       mixed_precision=False,
                                       compile_model=False,
                                       clip_grad=0.0)
        
        
        self.dataset_cfg = DynamicConfig(SMTGraghDataset,
                                        data_dir=train_data_dir,
                                        batch_size=16,  # 增加批次大小
                                        drop_last=False,
                                        shuffle=True,
                                        max_files_in_memory=5,  # 增加文件缓存
                                        preload_files=True)

        
        self.lr_scheduler_cfg = DynamicConfig(LRScheduler,
                                              peak_lr=2e-4,
                                              min_lr=1e-7,
                                              warmup_count=10,
                                              window_size=10,
                                              patience=10,
                                              decay_rate=0.5)
        
        self.constants = {
            "n_epochs": 100,
            "moving_avg": 100,
            "eval_interval": 2
        }

    def start(self, checkpoint: str = None) -> Trainer:
        rprint(f"[#00ff00]--- Start Experiment \"{self.comments}\" ---[/#00ff00]")

        # 创建不同的数据集实例
        print("正在创建训练数据集...")
        self.dataset_cfg.set_name = "all"
        train_set = self.dataset_cfg.build()
        
        print("正在创建验证数据集...")
        self.dataset_cfg.data_dir = self.eval_data_dir
        eval_set = self.dataset_cfg.build()

        model = self.model_cfg.build()
        model.initialize()

        if torch.cuda.device_count() > 1:
            print(f"使用{torch.cuda.device_count()}个GPU进行训练")
            model = torch.nn.DataParallel(model)
        model.to(DEVICE)

        # checkpoint 加载逻辑
        if checkpoint and isinstance(checkpoint, str) and os.path.exists(checkpoint):
            rprint(f"[blue]Loading checkpoint from: {checkpoint}[/blue]")
            try:
                if hasattr(model, 'module'):
                    model.module.loadFrom(checkpoint)
                else:
                    model.loadFrom(checkpoint)
                rprint(f"[green]Successfully loaded checkpoint[/green]")
            except Exception as e:
                rprint(f"[red]Failed to load checkpoint: {e}[/red]")
                raise

        # 设置优化器
        if hasattr(model, 'module'):
            self.lr_scheduler_cfg.optimizer = model.module.optimizer
        else:
            self.lr_scheduler_cfg.optimizer = model.optimizer

        lr_scheduler = self.lr_scheduler_cfg.build()

        trainer_kwargs = {"train_set": train_set, "eval_set": eval_set, "model": model, "lr_scheduler": lr_scheduler}
        trainer_kwargs.update(self.constants)

        # 创建实验目录
        now_str = datetime.now().strftime("%y%m%d_%H%M%S")
        dataset_name = trainer_kwargs["train_set"].__class__.__name__

        if hasattr(model, 'module'):
            model_name = model.module.__class__.__name__
        else:
            model_name = model.__class__.__name__

        save_dir = f"Runs/{dataset_name}/{model_name}/{now_str}/"
        log_dir = save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(log_dir, "model_arch.txt"), "w") as f:
            f.write(str(model))

        with open(os.path.join(log_dir, "comments.txt"), "w") as f:
            f.write(f"{self.comments}.\n{self.__str__()}")

        rprint(f"[blue]Save directory: {save_dir}.[/blue]")
        rprint(f"[blue]Log directory: {log_dir}.[/blue]")

        trainer_kwargs["log_dir"] = log_dir
        trainer_kwargs["save_dir"] = save_dir

        trainer = Trainer(**trainer_kwargs)
        trainer.start()

        # 测试阶段
        model_for_test = model.module if hasattr(model, 'module') else model
        if hasattr(model_for_test, 'state_dict') and self.is_test:

            
            rprint(f"[blue]Training done. Start testing.[/blue]")
            evaluate_model(
                model_path=save_dir,
                data_dir=self.test_data_dir,
                output_dir=save_dir
            )
            
            rprint(f"[blue]Testing done. Reports saved to: {os.path.join(log_dir, 'test_report.csv')}.[/blue]")

        return trainer
