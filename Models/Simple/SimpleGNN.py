import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, LayerNorm
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool
from typing import *
from Models.BaseModel import BaseModel




class SimpleGNN(BaseModel):
    def __init__(self, node_features_dim, 
                #  edge_features_dim, 
                 num_solvers,
                 optimizer_cls,
                 optimizer_args,
                 mixed_precision: bool = False, # 是否混合精度计算
                 compile_model: bool = False, # 是否编译模型加速运行
                 clip_grad: float = 0.0, # 是否进行梯度裁剪
                 hidden_dim=64, 
                 num_layers=1,
                 solver_embed_dim=16,
                 use_solver_attention=True,
                 ):
        super(SimpleGNN, self).__init__(
            optimizer_cls=optimizer_cls,
            optimizer_args=optimizer_args,
            mixed_precision=mixed_precision,
            compile_model=compile_model,
            clip_grad=clip_grad
        )

        self.use_solver_attention = use_solver_attention

        # 节点编码
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features_dim, hidden_dim), # 20->64
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # 图卷积


        # Solver编码 - 使用嵌入层
        self.solver_embedding = nn.Embedding(num_solvers, solver_embed_dim)
        
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for _ in range(num_layers):
            conv = GATConv(hidden_dim, hidden_dim//4, heads=4, dropout=0.1)
            self.conv_layers.append(conv)
            self.norm_layers.append(LayerNorm(hidden_dim))


        # Solver注意力机制（可选）
        if self.use_solver_attention:
            self.graph_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            self.solver_proj = nn.Linear(solver_embed_dim, hidden_dim)
            self.solver_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, 
                num_heads=4, 
                dropout=0.1,
                batch_first=True
            )
            self.solver_proj = nn.Linear(solver_embed_dim, hidden_dim)

        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )


        if self.use_solver_attention:
            # 使用注意力时，输出是hidden_dim + solver_embed_dim
            fusion_input_dim = hidden_dim + solver_embed_dim
        else:
            # 不使用注意力时，直接拼接
            fusion_input_dim = hidden_dim * 2 + solver_embed_dim

        self.feature_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )


        # 预测器
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加 dropout 提高泛化能力
            nn.Linear(hidden_dim//2, 1)
        )


        self.train_loss_names = ['Train_MSE']
        self.eval_loss_names = ["Eval_MSE"]
        self.loss_fn = nn.MSELoss()
        # optimizer_params = {"lr": 0.001} if optimizer_params is None else optimizer_params
        # # self.parameters() 返回需要优化权重 optimizer_params优化器的参数
        # self.optimizer = optimizer(self.parameters(), **optimizer_params) 
        

    def forward(self, x, edge_index, solver_ids, batch=None):
        # 节点编码
        x = self.node_encoder(x)
        
        # 图卷积层
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            residual = x
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            
            # 残差连接（如果需要的话）
            if i > 0:  # 从第二层开始添加残差连接
                x = x_new + residual
            else:
                x = x_new
        
        # 全局池化
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)

        # Solver编码
        solver_embed = self.solver_embedding(solver_ids)  # [batch_size, solver_embed_dim]
        
        # 使用注意力机制融合solver信息和图特征
        if self.use_solver_attention:
            # 将图特征投影到统一维度
            graph_features = torch.cat([x_mean, x_max], dim=1)  # [batch_size, hidden_dim*2]
            graph_proj = self.graph_proj(graph_features).unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # 将solver信息投影到相同维度
            solver_proj = self.solver_proj(solver_embed).unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # 使用注意力机制 - 现在所有维度都是hidden_dim
            attended_features, _ = self.solver_attention(
                query=solver_proj,
                key=graph_proj,
                value=graph_proj
            )
            attended_features = attended_features.squeeze(1)  # [batch_size, hidden_dim]
            
            # 特征融合
            x_global = torch.cat([attended_features, solver_embed], dim=1)
        else:
            # 简单拼接
            x_global = torch.cat([x_mean, x_max, solver_embed], dim=1)
        
        x_fused = self.feature_fusion(x_global)
        
        # 预测
        return self.predictor(x_fused)
    
    def trainStep(self, data_dict) -> Tuple[dict[str, Any], dict[str, Any]]:
        # 获取输入和标签
        x = data_dict['x'].to(self.device)
        edge_index = data_dict['edge_index'].to(self.device)
        batch = data_dict['batch'].to(self.device)
        y = data_dict['y'].to(self.device)
        solver_name = data_dict['solver']


        # 将solver名称转换为ID
        if isinstance(solver_name, (list, tuple)) and len(solver_name) > 0 and isinstance(solver_name[0], str):
            solver_ids = self._convert_solver_names_to_ids(solver_name)
        elif isinstance(solver_name, str):
            solver_ids = self._convert_solver_names_to_ids([solver_name])
        else:
            solver_ids = solver_name.to(self.device) if hasattr(solver_name, 'to') else solver_name

        
        pred = self(x, edge_index, solver_ids, batch).squeeze(-1)
        loss = self.loss_fn(pred, y)

        # 反向传播与优化
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        loss_dict = {'Train_MSE': loss.item()}
        output_dict = {'pred': pred.detach()}

        return loss_dict, output_dict
    
    def evalStep(self, data):

        # 获取输入和标签
        # 首先将所有数据移动到模型所在的设备
        x = data['x'].to(self.device)
        edge_index = data['edge_index'].to(self.device)
        batch = data['batch'].to(self.device)
        y = data['y'].to(self.device)
        solver_name = data['solver']
        
        # 将solver名称转换为ID
        if isinstance(solver_name, (list, tuple)) and len(solver_name) > 0 and isinstance(solver_name[0], str):
            solver_ids = self._convert_solver_names_to_ids(solver_name)
        elif isinstance(solver_name, str):
            solver_ids = self._convert_solver_names_to_ids([solver_name])
        else:
            solver_ids = solver_name.to(self.device) if hasattr(solver_name, 'to') else solver_name

        with torch.no_grad():
            pred = self(x, edge_index, solver_ids, batch).squeeze(-1)
            loss = self.loss_fn(pred, y)
        
        loss_dict = {'Eval_MSE': loss.item()}
        output_dict = {'pred': pred.detach()}

        return loss_dict, output_dict
    

    def _convert_solver_names_to_ids(self, solver_names):
        """
        将solver名称转换为ID
        """
        solver_to_id = {
            'z3': 0,
            'yices-2.6.2': 1,
            'yices-2.6.5': 2,
            'cvc4': 3,
            'cvc5': 4,
            'mathsat': 5,
        }
        
        if isinstance(solver_names, str):
            solver_names = [solver_names]
            
        ids = [solver_to_id.get(name, 0) for name in solver_names]
        return torch.tensor(ids, device=self.device)
    
    @property
    def device(self):
        """获取模型所在设备"""
        return next(self.parameters()).device
            


