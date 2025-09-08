import os
# 设置CUDA内存分配策略
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from Experiment import Experiment

if __name__ == '__main__':

    lr = 1e-3
    train_data_dir = "/root/autodl-tmp/project_gnn_original_data/FPV/train_split"
    eval_data_dir = "/root/autodl-tmp/project_gnn_original_data/FPV/eval_split"
    test_data_dir = "/root/autodl-tmp/project_gnn_original_data/FPV/test_split"

    experiment = Experiment(
        comments=f"test whole project with peak_lr={lr}, epoch=100", 
        train_data_dir=train_data_dir,
        eval_data_dir=eval_data_dir,
        test_data_dir=test_data_dir,
        is_test=True
    )
    experiment.lr_scheduler_cfg.peak_lr = lr
    # experiment.dataset_cfg.set_name = ""
    experiment.constants["n_epochs"] = 1

    experiment.start()