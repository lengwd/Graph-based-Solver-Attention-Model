import torch
from .BaseDataset import BaseDataset
from .DatasetUtils import * 
from typing import *
from torch_geometric.data import Batch

import os
from datetime import datetime



class SMTGraghDataset(BaseDataset):
    def __init__(self,
                 data_dir,
                 set_name: Literal["train", "eval", "test", "debug"],
                 batch_size: int,
                 drop_last: bool = False,
                 shuffle: bool = False,
                 ):
        
        super().__init__(batch_size, drop_last, shuffle)

        self.set_name = set_name  # 记录数据集类型
        # self.device = DEVICE      # 记录设备（CPU/GPU）
        self.graph_list = []    

        # 1. 从文件加载所有.pt格式的数据
        self._load_data(data_dir)
        
        # 2. 根据set_name划分数据集（类似MNIST的处理逻辑）
        self._split_dataset()
        
        # # 3. 将数据迁移到指定设备
        # self._move_to_device()


    def _load_data(self, data_dir: List[str]):
        """从指定目录列表加载所有.pt文件中的图数据"""
        # 确保data_dir是列表格式
        if isinstance(data_dir, str):
            data_dir = [data_dir]
        
        # 遍历每个数据目录
        for dir_path in data_dir:
            if not os.path.exists(dir_path):
                print(f"警告: 数据目录不存在，跳过: {dir_path}")
                continue
            
            print(f"正在加载目录: {dir_path}")
            dir_loaded_count = 0
            
            # 遍历目录下所有.pt文件
            for file_name in os.listdir(dir_path):
                if file_name.endswith(".pt"):  # 确保只处理.pt文件
                    file_path = os.path.join(dir_path, file_name)
                    try:
                        # 加载文件中的数据（假设每个文件存储一个图数据列表）
                        data_list = torch.load(file_path, weights_only=False)
                        # 扩展到总列表中（确保data_list是可迭代的图数据）
                        self.graph_list.extend(data_list)
                        dir_loaded_count += len(data_list)
                        print(f"  成功加载 {file_name}，新增 {len(data_list)} 个图数据")
                    except Exception as e:
                        print(f"  加载 {file_name} 失败: {str(e)}")
            
            print(f"目录 {dir_path} 加载完成，共加载 {dir_loaded_count} 个图数据")
        
        # 检查是否加载到数据
        if not self.graph_list:
            raise RuntimeError(f"在所有指定目录中未找到有效数据: {data_dir}")
        
        print("数据集结构", self.graph_list[0], self.graph_list[0].y.item())
        
        # 初始化总样本数
        self.n_samples = len(self.graph_list)
        print(f"数据加载完成，共 {self.n_samples} 个图数据")

    def _split_dataset(self):
        """根据set_name划分数据集"""
        # 记录原始数据长度，用于划分
        original_len = self.n_samples
        
        match self.set_name:
            case "train":
                # 取前90%作为训练集
                self.n_samples = int(original_len * 0.7)
                self.graph_list = self.graph_list[:self.n_samples]
                print(f"划分训练集: {self.n_samples} 个样本")
            
            case "eval":
                # 取后10%作为验证集（基于训练集划分）
                eval_start = int(original_len * 0.7)
                eval_end = int(original_len * 0.8)
                self.graph_list = self.graph_list[eval_start:eval_end]
                self.n_samples = len(self.graph_list)
                print(f"划分验证集: {self.n_samples} 个样本")
            
            case "test":
                test_start = int(original_len * 0.8)

                self.graph_list = self.graph_list[test_start:]
                self.n_samples = len(self.graph_list)
                # 测试集使用全部数据（假设test目录下的数据已单独准备）
                print(f"使用完整测试集: {self.n_samples} 个样本")
            
            case "debug":
                # 调试用小数据集（取前300个）
                self.n_samples = min(300, original_len)
                self.graph_list = self.graph_list[:self.n_samples]
                print(f"划分调试集: {self.n_samples} 个样本")
            
            case _:
                raise ValueError(f"未知的数据集类型: {self.set_name}")


    def _move_to_device(self):
        """将图数据迁移到指定设备（CPU/GPU）"""
        # 假设图数据是PyTorch Geometric的Data对象或包含tensor的字典
        # 遍历所有图，将其中的tensor迁移到设备
        for i in range(self.n_samples):
            # 如果是PyTorch Geometric的Data对象
            if hasattr(self.graph_list[i], 'to'):
                self.graph_list[i] = self.graph_list[i].to(self.device)
            # 如果是字典（键为'tensor_name'，值为tensor）
            elif isinstance(self.graph_list[i], dict):
                for key, value in self.graph_list[i].items():
                    if isinstance(value, torch.Tensor):
                        self.graph_list[i][key] = value.to(self.device)
            else:
                raise TypeError(f"不支持的数据类型: {type(self.graph_list[i])}")
        print(f"数据已迁移到设备: {self.device}")


    def __getitem__(self, idx):
        """重写获取样本的方法（返回字典格式，统一接口）"""
        # 注意：这里的idx是批次索引（继承自BaseDataset的迭代逻辑）

        start = (idx - 1) * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        batch_graphs = self.graph_list[start:end]
        # batch_ys = [data.y for data in batch_graphs]

        
        batch = Batch.from_data_list(batch_graphs)
        
        return {
                'x': batch.x,
                'edge_index': batch.edge_index,
                'batch': batch.batch,
                'y': batch.y,
                'solver': batch.solver,
                'benchmark_path': batch.benchmark_path
            }

    
    def __len__(self):
        return len(self.graph_list)
    
    @staticmethod
    def collate_fn(batch):
        from torch_geometric.data import Batch
        return Batch.from_data_list(batch)
    

if __name__ == "__main__":
    cc_bin_dir = "/root/autodl-tmp/project_gnn_original_data/formal_verification_data/CC/bin"
    device = "cpu"

    CCGraghDataset = SMTGraghDataset(
        data_dir = cc_bin_dir,
        set_name = "train",
        batch_size = 32,
        drop_last= False,
        shuffle= True
    )

    print(CCGraghDataset[2])


    print(len(CCGraghDataset))

        
    


