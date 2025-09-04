import torch
import torch.nn as nn
from typing import Any, Tuple
import os


class BaseModel(nn.Module):
    """
    BaseModel defines a model format that compatible with many different models.

    Most importantly, the forwardBackward function returns a dictionary, this is helpful for unifying the training code
    when you want to use different models for different datasets in your experiments.
    """

    def __init__(self,
                 optimizer_cls=None, # 优化器类别
                 optimizer_args=None, # 优化器参数
                 mixed_precision: bool = False, # 是否混合精度计算
                 compile_model: bool = False, # 是否编译模型加速运行
                 clip_grad: float = 0.0 # 是否进行梯度裁剪
                 ):
        super(BaseModel, self).__init__()

        self.train_loss_names = ["Train_loss"]
        self.eval_loss_names = ["Eval_loss"]
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer_cls = optimizer_cls
        self.optimizer_args = optimizer_args
        self.mixed_precision = mixed_precision
        self.compile_model = compile_model

        try:
            self.scaler = torch.amp.GradScaler() if mixed_precision else None
        except AttributeError:
            self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        
        self.clip_grad = clip_grad

    def initialize(self) -> None:
        """
        Initialize the optimizer for the model.
        :param optimizer_cls: The optimizer class to use (e.g., torch.optim.Adam).
        :param optimizer_args: A dictionary of arguments to pass to the optimizer constructor.
        :return:
        """

        if self.compile_model:
            torch.set_float32_matmul_precision('high')
            self.compile()

        if self.optimizer_cls is None or self.optimizer_args is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        else:
            self.optimizer = self.optimizer_cls(self.parameters(), **self.optimizer_args)
    
    # 可以直接通过 self.lr 访问学习率
    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']
    

    def trainStep(self, data_dict) -> Tuple[dict[str, Any], dict[str, Any]]:
        
        # 如果混合精度计算
        if self.mixed_precision:
            with torch.autocast(device_type=data_dict['data'].device, dtype=torch.float16):
                output = self(data_dict['data'])
                loss = self.loss_fn(output, data_dict['target'])

            # Backward pass with AMP
            self.scaler.scale(loss).backward()

            # Gradient clipping with AMP (if specified)
            if self.clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            # Optimizer step with AMP
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            output = self(data_dict['data'])
            loss = self.loss_fn(output, data_dict['target'])
            loss.backward()

            # Standard gradient clipping (if specified)
            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)

            # Optimizer step
            self.optimizer.step()

         # Zero the gradients
        self.optimizer.zero_grad()


        return {"Train_loss": loss.item()}, {"output": output.detach()}



    def evalStep(self, data_dict) -> Tuple[dict[str, Any], dict[str, Any]]:
        with torch.no_grad():
            output = self(data_dict['data']).detach()
            loss = self.loss_fn(output, data_dict['target']).item()
        return {"Eval_loss": loss}, {"output": output.detach()}

    
    def saveTo(self, path: str):
        torch.save(self.state_dict(), path)

    def loadFrom(self, path: str):

        """加载模型检查点"""
        if path is None:
            raise ValueError("Checkpoint path cannot be None")
        
        if not isinstance(path, str):
            raise ValueError(f"Checkpoint path must be a string, got {type(path)}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        print(f"Loading model from: {path}")
        state_dict = torch.load(path, weights_only=False) # 表示允许加载非张量类型的额外数据（默认行为）
        current_state_dict = self.state_dict()

        # Filter and handle mismatched parameters
        mis_matched_keys = set() # 存储形状不匹配的参数名
        loadable_state_dict = {} # 存储可加载的参数（名称和形状都匹配）
        for param_name, param_value in state_dict.items():
            if param_name in current_state_dict:
                if current_state_dict[param_name].size() == param_value.size():
                    loadable_state_dict[param_name] = param_value
                else:
                    mis_matched_keys.add(param_name)
                    print(
                        f"Warning! Parameter '{param_name}' expect size {current_state_dict[param_name].shape} but got {param_value.shape}. Skipping.")
            else:
                print(f"Unexpected parameter '{param_name}''. Skipping.")

        # Load filtered parameters
        self.load_state_dict(loadable_state_dict, strict=False)

        # Check for missing parameters
        for param_name in current_state_dict.keys():
            if param_name not in loadable_state_dict and param_name not in mis_matched_keys:
                print(f"Missing parameter '{param_name}' in model '{self.__class__.__name__}'.")