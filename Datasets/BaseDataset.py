from .DatasetUtils import *


class BaseDataset():
    """
    JimmyDataset defines a dataset format that compatible with many different datasets.

    Most importantly, the __getitem__ function returns a dictionary, this is helpful for unifying the training code
    when you want to use different datasets for different models in your experiments.

    """
    def __init__(self, batch_size: int, drop_last: bool = False, shuffle: bool = False):
        """
        :param batch_size: number of samples per batch
        :param drop_last: if True, drop the last batch if it is smaller than batch_size
        :param shuffle: if True, shuffle the dataset at the beginning of each epoch
        """
        rprint("[blue]Initializing dataset[/blue]")
        self.batch_size = batch_size
        self.n_samples = 100
        self.drop_last = drop_last
        self.shuffle = shuffle

    @property # 可以直接通过 dataset.n_batchs 来访问batch的数量
    def n_batches(self) -> int:
        """ Because n_samples may be set later, we need to recalculate n_batches every time """
        if self.drop_last:
            return self.n_samples // self.batch_size
        else:
            return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __len__(self) -> int:
        return self.n_samples


    def __iter__(self): # 构建 一个迭代器
        self.iter_idx = 0 
        if self.shuffle:
            # self._indices 内部使用的索引
            self._indices = torch.randperm(self.n_samples)  # 用于生成一个0 到 n-1 的随机排列的整数张量
        else:
            self._indices = torch.arange(self.n_samples) # 用于生成一个从 0 到 n-1 的连续整数张量
        return self


    def __next__(self): # 迭代的逻辑
        if self.iter_idx >= self.n_batches:
            raise StopIteration
        self.iter_idx += 1
        return self.__getitem__(self.iter_idx)


    def __getitem__(self, idx):
        start = (idx - 1) * self.batch_size
        end = min(start + self.batch_size, self.n_samples)
        indices = self._indices[start:end]


        return {
            'indices': indices,
            'input': torch.randn(len(indices), 10),
            'target': torch.randint(0, 2, (len(indices),))
        }

# 简单测试代码
if __name__ == "__main__":
    # 1. 创建测试实例（批次大小20，不丢弃最后一批，不打乱）
    dataset = BaseDataset(batch_size=20, drop_last=False, shuffle=False)
    print(f"测试1：不打乱模式，总样本数={len(dataset)}，总批次={dataset.n_batches}")
    
    # 遍历前2批并打印信息
    for i, batch in enumerate(dataset):
        if i >= 2:  # 只看前2批
            break
        print(f"第{i+1}批 - 样本数：{len(batch['indices'])}，索引：{batch['indices'].tolist()[:5]}...")

    # 2. 测试打乱模式
    dataset_shuffle = BaseDataset(batch_size=20, shuffle=True)
    print(f"\n测试2：打乱模式，总批次={dataset_shuffle.n_batches}")
    first_batch_indices = next(iter(dataset_shuffle))['indices']
    second_batch_indices = next(iter(dataset_shuffle))['indices']  # 重新迭代获取新顺序
    print(f"第一次迭代首批索引：{first_batch_indices.tolist()[:5]}...")
    print(f"第二次迭代首批索引：{second_batch_indices.tolist()[:5]}...")
    print(f"打乱是否生效：{not torch.allclose(first_batch_indices, second_batch_indices)}")

    # 3. 测试drop_last参数
    dataset_drop = BaseDataset(batch_size=30, drop_last=True)
    print(f"\n测试3：drop_last=True，总批次={dataset_drop.n_batches}（预期3）")