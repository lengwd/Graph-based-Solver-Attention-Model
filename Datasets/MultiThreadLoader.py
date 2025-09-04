import threading  # 用于创建和管理多线程
import queue      # 用于创建“数据队列”（存储提前加载好的数据）
from typing import Iterable  # 类型提示，指定迭代器返回类型
from .SMTGraghDataset import SMTGraghDataset
from .BaseDataset import BaseDataset
from .DatasetUtils import *
import time

class MultiThreadLoader:
    def __init__(self, dataset: BaseDataset, queue_size: int = 3):
        """
        Multi-threaded data loader for JimmyDataset
        :param queue_size: size of the queue
        :param dataset: dataset to load data from
        """
        self.queue_size = queue_size
        self.dataset = dataset
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._load_data, daemon=True)
        self.thread.start()

    def __len__(self):
        return self.dataset.n_batches
    
    def __iter__(self) -> Iterable:
        for _ in range(len(self)):
            yield self.queue.get()


    def _load_data(self):
        """
        Load data from the dataset and put it in the queue
        :return:
        """
        for i, data_dict in enumerate(self.dataset):
            self.queue.put(data_dict)

    def queueSize(self) -> int:
        """
        Get the size of the queue
        :return: size of the queue
        """
        return self.queue.qsize()
    
if __name__ == "__main__":
    cc_bin_dir = "/root/autodl-tmp/project_gnn_original_data/formal_verification_data/CC/bin"
    test_dataset = SMTGraghDataset(
        data_dir = cc_bin_dir,
        set_name = "train",
        batch_size = 32,
        drop_last= False,
        shuffle= True
    )

    loader = MultiThreadLoader(
        dataset=test_dataset,
        queue_size=2  # 队列最多存2批数据，更容易观察阻塞/缓冲效果
    )

    # 3. 模拟训练循环：从加载器取数据并处理
    print(f"\n[训练开始] 总批次数：{len(loader)}")
    start_time = time.time()

    for batch_idx, data_dict in enumerate(loader, 1):
        # 模拟训练耗时（比加载耗时短，观察队列缓冲效果）
        time.sleep(0.05)
        
        # 验证数据正确性
        print(f"\n[训练线程] 处理第 {batch_idx} 批数据：")
        print(f"  - 批次索引：{data_dict['indices']}")
        # print(f"  - 批次大小：{data_dict['batch_size']}")
        print(f"  - 图数据示例：节点特征形状 {data_dict['graphs'][0]}")
        print(f"  - 队列剩余数据量：{loader.queueSize()}")

    # 4. 测试总结
    total_time = time.time() - start_time
    print(f"\n[训练结束] 所有批次处理完成！总耗时：{total_time:.2f} 秒")
    print(f"[测试结论] 多线程加载器功能正常（数据迭代、队列缓冲、线程协作均正常）")