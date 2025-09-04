import os
# 设置CUDA内存分配策略
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from Experiment import Experiment

if __name__ == '__main__':

    lr = 1e-3
    data_dir = [
        '/root/autodl-tmp/project_gnn_original_data/formal_verification_data/CC/bin',
        '/root/autodl-tmp/project_gnn_original_data/formal_verification_data/FPV/bin',
        '/root/autodl-tmp/project_gnn_original_data/formal_verification_data/UNR/bin'
    ]
    experiment = Experiment(comments=f"test whole project with peak_lr={lr}, epoch=100", data_dir=data_dir, is_test=True)
    experiment.lr_scheduler_cfg.peak_lr = lr
    # experiment.dataset_cfg.set_name = ""
    experiment.constants["n_epochs"] = 100


    experiment.start()
