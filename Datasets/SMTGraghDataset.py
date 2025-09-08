import torch
from .BaseDataset import BaseDataset
from .DatasetUtils import * 
from typing import *
from torch_geometric.data import Batch
import os
import gc
import psutil
from datetime import datetime
import threading
from collections import OrderedDict

class SMTGraghDataset(BaseDataset):
    def __init__(self,
                 data_dir,
                 set_name: Literal["all", "train", "eval", "test", "debug"],
                 batch_size: int,
                 drop_last: bool = False,
                 shuffle: bool = False,
                 max_files_in_memory: int = 3,  # 同时在内存中保持的文件数
                 preload_files: bool = True,    # 是否预加载文件
                 ):
        
        super().__init__(batch_size, drop_last, shuffle)

        self.set_name = set_name
        self.max_files_in_memory = max_files_in_memory
        self.preload_files = preload_files
        
        # 文件级缓存
        self.file_cache = OrderedDict()  # {file_idx: data_list}
        self.file_access_count = {}      # 文件访问计数
        
        # 索引信息
        self.file_paths = []
        self.file_sample_counts = []
        self.sample_to_file_map = []
        
        # 构建索引
        self._build_file_index(data_dir)
        self._split_dataset()
        
        # 预加载最常用的文件
        if self.preload_files:
            self._preload_frequent_files()

    def _build_file_index(self, data_dir):
        """快速构建文件索引"""
        if isinstance(data_dir, str):
            data_dir = [data_dir]
        
        total_samples = 0
        
        for dir_path in data_dir:
            if not os.path.exists(dir_path):
                continue
            
            print(f"正在扫描目录: {dir_path}")
            
            for file_name in sorted(os.listdir(dir_path)):  # 排序确保一致性
                if file_name.endswith(".pt"):
                    file_path = os.path.join(dir_path, file_name)
                    try:
                        # 快速获取样本数量
                        sample_count = self._get_file_sample_count_fast(file_path)
                        
                        self.file_paths.append(file_path)
                        self.file_sample_counts.append(sample_count)
                        
                        # 建立样本到文件的映射
                        for i in range(sample_count):
                            self.sample_to_file_map.append((len(self.file_paths) - 1, i))
                        
                        total_samples += sample_count
                        print(f"  扫描 {file_name}，包含 {sample_count} 个样本")
                        
                    except Exception as e:
                        print(f"  扫描 {file_name} 失败: {str(e)}")
        
        self.n_samples = total_samples
        print(f"文件扫描完成，共发现 {self.n_samples} 个样本")

    def _get_file_sample_count_fast(self, file_path: str) -> int:
        """快速获取文件样本数量"""
        try:
            # 使用 map_location='cpu' 避免GPU内存占用
            data = torch.load(file_path, map_location='cpu', weights_only=False)
            count = len(data) if isinstance(data, list) else 1
            del data
            return count
        except:
            return 0

    def _split_dataset(self):
        """数据集划分"""
        original_len = self.n_samples
        
        match self.set_name:
            case "all":
                self.sample_indices = list(range(0, original_len))
            case "train":
                end_idx = int(original_len * 0.7)
                self.sample_indices = list(range(0, end_idx))
            case "eval":
                start_idx = int(original_len * 0.7)
                end_idx = int(original_len * 0.8)
                self.sample_indices = list(range(start_idx, end_idx))
            case "test":
                start_idx = int(original_len * 0.8)
                self.sample_indices = list(range(start_idx, original_len))
            case "debug":
                self.sample_indices = list(range(min(300, original_len)))
        
        self.n_samples = len(self.sample_indices)
        print(f"划分{self.set_name}集: {self.n_samples} 个样本")

    def _preload_frequent_files(self):
        """预加载最常访问的文件"""
        # 统计每个文件在当前数据集中的样本数量
        file_usage = {}
        for idx in self.sample_indices:
            file_idx, _ = self.sample_to_file_map[idx]
            file_usage[file_idx] = file_usage.get(file_idx, 0) + 1
        
        # 按使用频率排序
        sorted_files = sorted(file_usage.items(), key=lambda x: x[1], reverse=True)
        
        # 预加载前几个最常用的文件
        preload_count = min(self.max_files_in_memory, len(sorted_files))
        print(f"预加载 {preload_count} 个最常用文件...")
        
        for i in range(preload_count):
            file_idx, usage_count = sorted_files[i]
            self._load_file(file_idx)
            print(f"  预加载文件 {i+1}/{preload_count}: {os.path.basename(self.file_paths[file_idx])} (使用 {usage_count} 次)")

    def _load_file(self, file_idx: int):
        """加载指定文件到缓存"""
        if file_idx in self.file_cache:
            # 更新访问顺序
            self.file_cache.move_to_end(file_idx)
            return self.file_cache[file_idx]
        
        # 检查缓存大小，必要时清理
        while len(self.file_cache) >= self.max_files_in_memory:
            oldest_file_idx = next(iter(self.file_cache))
            del self.file_cache[oldest_file_idx]
            gc.collect()
        
        # 加载文件
        file_path = self.file_paths[file_idx]
        try:
            print(f"加载文件: {os.path.basename(file_path)}")
            data_list = torch.load(file_path, map_location='cpu', weights_only=False)
            self.file_cache[file_idx] = data_list
            return data_list
        except Exception as e:
            print(f"加载文件失败 {file_path}: {e}")
            raise

    def __getitem__(self, batch_idx):
        """获取批次数据"""
        start = (batch_idx - 1) * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        
        batch_samples = []
        
        # 按文件分组批次中的样本，减少文件加载次数
        file_samples = {}  # {file_idx: [sample_indices]}
        
        for i in range(start, end):
            global_idx = self.sample_indices[i]
            file_idx, sample_idx = self.sample_to_file_map[global_idx]
            
            if file_idx not in file_samples:
                file_samples[file_idx] = []
            file_samples[file_idx].append(sample_idx)
        
        # 批量从每个文件获取样本
        for file_idx, sample_indices in file_samples.items():
            data_list = self._load_file(file_idx)
            for sample_idx in sample_indices:
                batch_samples.append(data_list[sample_idx])
        
        # 创建批次
        batch = Batch.from_data_list(batch_samples)
        
        return {
            'x': batch.x,
            'edge_index': batch.edge_index,
            'batch': batch.batch,
            'y': batch.y,
            'solver': batch.solver,
            'benchmark_path': batch.benchmark_path
        }

    def __len__(self):
        return self.n_samples
