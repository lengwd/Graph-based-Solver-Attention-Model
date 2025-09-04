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
    """
    This is an example of an experiment class that defines the hyperparameters and constants for the experiment.
    For other type of experiments, or your customized trainer, you should write a new experiment class to accommodate
    the new set of hyperparameters and constants.
    """

    def __init__(self, comments: str, data_dir: str, is_test: bool = False):
        self.comments = comments
        self.is_test = is_test

        self.model_cfg = DynamicConfig(SimpleGNN,
                                       node_features_dim = 20,
                                    #    edge_features_dim = 5,
                                       num_solvers=6,
                                       optimizer_cls=torch.optim.Adam,
                                       optimizer_args={"lr": 1e-5},
                                       mixed_precision=False,
                                       compile_model=False,
                                       clip_grad=0.0)
        
        self.dataset_cfg = DynamicConfig(SMTGraghDataset,
                                         data_dir=data_dir,
                                         batch_size=8,
                                         drop_last=False,
                                         shuffle=True)
        
        # The default hyperparameters for the experiment.
        self.lr_scheduler_cfg = DynamicConfig(LRScheduler,
                                              peak_lr=2e-4,
                                              min_lr=1e-7,
                                              warmup_count=10,
                                              window_size=10,
                                              patience=10,
                                              decay_rate=0.5)
        
        # Other constants for the experiment.
        self.constants = {
            "n_epochs": 100,
            "moving_avg": 100,
            "eval_interval": 2
        }

    def __str__(self):
        return (f"Experiment{{\n"
                f"\tdataset={self.dataset_cfg}\n"
                f"\tmodel={self.model_cfg}\n"
                f"\tlr_scheduler={self.lr_scheduler_cfg}\n"
                f"\tconstants={self.constants}\n}}")
    
    def __repr__(self):
        return self.__str__()
    
    def start(self, checkpoint: str = None) -> Trainer:
        """
        Start the experiment with the given comments.
        :param checkpoint: The checkpoint to load the model from.
        :return: A `Trainer` object with amost everything during a training session.
        """
        rprint(f"[#00ff00]--- Start Experiment \"{self.comments}\" ---[/#00ff00]")

        self.dataset_cfg.set_name = "train"
        train_set = self.dataset_cfg.build()
        self.dataset_cfg.set_name = "eval"
        eval_set = self.dataset_cfg.build()

        model = self.model_cfg.build()

        model.initialize()

        if torch.cuda.device_count() > 1:
            print(f"使用{torch.cuda.device_count()}个GPU进行训练")
            model = torch.nn.DataParallel(model)
        model.to(DEVICE)

        
        # 改进的 checkpoint 加载逻辑
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
        elif checkpoint:
            rprint(f"[yellow]Warning: Invalid checkpoint path '{checkpoint}', skipping checkpoint loading[/yellow]")

        # 修改这里：根据是否使用DataParallel来获取optimizer
        if hasattr(model, 'module'):
            self.lr_scheduler_cfg.optimizer = model.module.optimizer
        else:
            self.lr_scheduler_cfg.optimizer = model.optimizer




        lr_scheduler = self.lr_scheduler_cfg.build()

        trainer_kwargs = {"train_set": train_set, "eval_set": eval_set, "model": model, "lr_scheduler": lr_scheduler}
        trainer_kwargs.update(self.constants)

        # Create Experiment directories
        now_str = datetime.now().strftime("%y%m%d_%H%M%S")
        dataset_name = trainer_kwargs["train_set"].__class__.__name__

        if hasattr(model, 'module'):
            model_name = model.module.__class__.__name__
        else:
            model_name = model.__class__.__name__


        save_dir = f"Runs/{dataset_name}/{model_name}/{now_str}/"
        log_dir = save_dir

        # Create directories if they do not exist
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

        # rprint(f"[blue]Training done. Start testing.[/blue]")
        # self.dataset_cfg.set_name = "test"
        # test_set = self.dataset_cfg.build()
        # test_losses = trainer.evaluate(test_set, compute_avg=False)

        # test_report = pd.DataFrame.from_dict(test_losses)
        # test_report.to_csv(os.path.join(log_dir, "test_report.csv"))

        # rprint(f"[blue]Testing done. Reports saved to: {os.path.join(log_dir, 'test_report.csv')}.[/blue]")


        # 修改测试部分的条件判断
        model_for_test = model.module if hasattr(model, 'module') else model
        if hasattr(model_for_test, 'state_dict') and self.is_test:
            rprint(f"[blue]Training done. Start testing.[/blue]")
            evaluate_model(
                model_path=save_dir,
                data_dir=self.dataset_cfg.data_dir,
                output_dir=save_dir
            )
            rprint(f"[blue]Testing done. Reports saved to: {os.path.join(log_dir, 'test_report.csv')}.[/blue]")

        return trainer   
