import torch


class MovingAvg():
    def __init__(self, window_size: int):
        self.window_size = window_size  # 滑动窗口的大小（要记录多少个最近的数据）
        # 创建一个存储数据的数组（用PyTorch的tensor），大小为window_size，值初始为0
        # dtype=torch.float32：数据类型是32位浮点数（用于存储小数）
        # device='cuda'：如果有GPU，就把数据存在GPU上（加速计算），没有的话可能会报错（可改为device='cpu'）
        self.values = torch.zeros(window_size, dtype=torch.float32, device='cuda')
        self.idx = 0  # 当前要更新的位置索引（从0开始）
        self.count = 0  # 已经记录的数据数量（不超过window_size）

    def update(self, value: float):
        # 把新数据存入数组的当前位置
        self.values[self.idx] = value
        # 更新索引：下一个数据要存在当前位置的下一个，到末尾后回到开头（循环利用数组）
        self.idx = (self.idx + 1) % self.window_size
        # 记录数据的数量：如果还没存满窗口大小，就加1；存满后保持不变
        self.count = min(self.count + 1, self.window_size)

    def get(self) -> float:
        # 计算已存数据的平均值：取数组中前count个数据（有效数据），求平均，再转换为Python的float类型
        return self.values[:self.count].mean().item()
    
    def __len__(self):
        return self.count  # 返回已存数据的数量