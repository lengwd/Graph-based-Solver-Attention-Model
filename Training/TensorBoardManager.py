from torch.utils.tensorboard import SummaryWriter  # TensorBoard的核心写入工具
from typing import Literal  # 用于限制参数只能是特定值（类型提示）
import os  # 用于处理文件路径和目录


class TensorBoardManager:
    def __init__(self, log_dir: str, tags: list[str] = None, value_types: list[str] = None):
        """
        初始化TensorBoard管理器
        :param log_dir: 保存TensorBoard日志的文件夹路径
        """
        # 如果日志文件夹不存在，就创建它
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 创建SummaryWriter对象，用于写入日志（日志会保存在log_dir文件夹）
        self.writer = SummaryWriter(log_dir=log_dir)
        
        # 用于记录已注册的"标签"和对应的日志类型（如scalar、figure）
        self.tag_registry = {}
        
        # 如果初始化时提供了tags和value_types，就批量注册它们
        if tags is not None and value_types is not None:
            # 检查tags和value_types的数量是否一致（一一对应）
            if len(tags) != len(value_types):
                raise ValueError("tags和value_types的长度必须相同")
            
            # 遍历tags和value_types，逐个注册
            for tag, value_type in zip(tags, value_types):
                self.register(tag, value_type)

    def register(self, tag: str, value_type: Literal['scalar', 'figure']):
        """
        注册一个新的标签（用于日志记录）
        :param tag: 要注册的标签名称（如"train_loss"）
        :param value_type: 日志的数据类型（只能是'scalar'或'figure'）
        """
        # 如果标签已经注册过，就报错（避免重复）
        if tag in self.tag_registry:
            raise ValueError(f"标签'{tag}'已经注册过了")
        
        # 根据value_type，为标签绑定对应的日志写入方法
        match value_type:
            case 'scalar':
                # 标量类型（如数字：loss值、准确率），绑定add_scalar方法
                self.tag_registry[tag] = self.writer.add_scalar
            case 'figure':
                # 图表类型（如matplotlib绘制的图），绑定add_figure方法
                self.tag_registry[tag] = self.writer.add_figure
            case _:
                # 如果不是支持的类型，就报错
                raise ValueError(f"不支持的数据类型'{value_type}'，只能是'scalar'或'figure'")
            
    def log(self, global_step: int, **values):
        """
        向TensorBoard写入日志
        :param global_step: 全局步骤数（如训练到第几步）
        :param values: 关键字参数，键是标签（tag），值是要记录的数据
        """
        # 遍历要写入的所有标签和对应的值
        for tag, value in values.items():
            # 调用该标签注册时绑定的方法（add_scalar或add_figure），写入数据
            self.tag_registry[tag](tag, value, global_step=global_step)