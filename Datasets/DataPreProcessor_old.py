import torch
from torch_geometric.data import Data
import pickle
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
from Datasets.YosysSMTtoTree import YosysSMTtoTree
import os
from datetime import datetime
import numpy as np


def smt_gnn_parse_to_pyg_data(features, solve_time=None, solver=None, result=None):
    "translate the format extract by gnn_parse.py to pyg data format"

    if len(features['node_features']) == 0:
        raise ValueError("节点特征为空")
    
    # 检查边索引
    if 'edge_index' not in features or len(features['edge_index']) == 0:
        raise ValueError("边索引为空或不存在")
    
    # 检查边索引的形状
    edge_index_array = features['edge_index']
    if len(edge_index_array) != 2:
        raise ValueError(f"边索引应该是2xN的数组，但得到了形状: {len(edge_index_array)}")


    x = torch.FloatTensor(features['node_features'])

    edge_index = torch.LongTensor(features['edge_index'])

    edge_attr = torch.FloatTensor(features["edge_attr"]) if len(features["edge_attr"]) > 0 else None

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    if solve_time is not None:
        # 对求解时间取对数，因为时间通常是长尾分布
        normalized_time = np.log(solve_time + 1e-8)  # 加小值避免log(0)
        data.y = torch.FloatTensor([normalized_time])

    if solver is not None:
        data.solver = solver
    
    if result is not None:
        data.result = result

    return data


def create_smt_dataset_from_csv(csv_file, focus_solver = 'z3', save_path=None, dataset_class=None, save_format='pickle'):
    """
    从CSV文件创建SMT数据集，支持进度条显示和多种保存格式
    
    Args:
        csv_file: CSV文件路径
        focus_solver: 目标求解器，默认为'z3'
        save_path: 保存路径，如果为None则不保存
        save_format: 保存格式，支持'pickle', 'torch', 'both'
    
    Returns:
        dataset: 处理后的数据集列表
    """

    # df = pd.read_csv(csv_file, names=['benchmark', 'solver', 'result', 'solve_time'], skiprows=1)
    df = pd.read_csv(csv_file, names=['benchmark', 'solver',  'solve_time'], skiprows=1)
    focus_data = df[df['solver'] == focus_solver]

    if focus_data.empty:
        print(f"警告：没有找到{focus_solver}求解器的数据")
        return []
    
    print(f"{focus_solver}数据条数: {len(focus_data)}")
    print(f"{focus_solver}求解状态分布:")
    # print(focus_data['result'].value_counts())
    print(f"{focus_solver}平均求解时间: {focus_data['solve_time'].mean():.2f}")


    benchmark_groups =  list(df.groupby("benchmark"))

    dataset = []
    successful_count = 0
    failed_count = 0
    
    print(f"Star processing {len(benchmark_groups)} benchmark files...")

    for benchmark_path, group in tqdm(benchmark_groups, desc='processing file', unit='file'):

        solver_result = group[group['solver']==focus_solver]

        if solver_result.empty:
            continue

        try:
            parser = YosysSMTtoTree()
            parser.parse_smt_file(benchmark_path)
            features = parser.extract_features()

            for _, row in solver_result.iterrows():
                current_solver_name = row["solver"]
                # solve_result = row["result"]
                solve_result = None
                solve_time = row['solve_time']

                data = smt_gnn_parse_to_pyg_data(features, solve_time, current_solver_name, solve_result)

                dataset.append(data)
                successful_count += 1

        except Exception as e:
            failed_count += 1
            tqdm.write(f"Error processing file {benchmark_path}: {e}")

            continue

    print(f"\n Procesing completed")
    print(f"Successful processing: {successful_count} files")
    print(f"Failed processing: {failed_count} files")
    print(f"Final dataset size: {len(dataset)}")

    if save_path is not None:
        _save_dataset(dataset, save_path, dataset_class, save_format, focus_solver)


    return dataset

def _save_dataset(dataset, save_path, dataset_class, save_format, solver_name):
        """
        Save the dataset in a format suitable for neural network training

        Args:
        dataset: dataset to be saved
        save_path: save path (without extension)
        save_format: save format ('pickle', 'torch', 'both')
        solver_name: solver name
        """
        try:
            # os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            if save_path and os.path.dirname(save_path):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # if solver_name in ["yices-2.6.2", "yices_2_6_2"]:
            #     solver_name = "yices_262"
            # if solver_name == ["yices-2.6.5", "yices_2_6_5"]:
            #     solver_name = "yices_265"
            
            solver_name_dict = {
                "yices-2.6.2": "yices_262",
                "yices_2_6_2": "yices_262",
                "yices-2.6.5": "yices_265",
                "yices_2_6_5": "yices_265",
                "Boolector-wrapped-sq": "boolector",
                "CVC4-2019-06-03-d350fe1-wrapped-sq": "cvc4",
                "Poolector-wrapped-sq": "poolector",
                "Q3B-wrapped-sq": "q3b",
                "UltimateEliminator+MathSAT-5.5.4-wrapped-sq":"mathsat",
                "z3-4.8.4-d6df51951f4c-wrapped-sq":"z3",
                "UltimateEliminator+Yices-2.6.1-wrapped-sq": "yices_262",
                "vampire-4.4-smtcomp-wrapped-sq": "vampire",
                "COLIBRI 20.5.25": "COLIBRI",
                "CVC4-sq-final": "cvc4",
                "MathSAT5": "mathsat",
                # "veriT",
                "smtinterpol-2.5-671-g6d0a7c6e": "smtinterpol",
                "z3-4.8.8": "z3_488",
                "Yices 2.6.2 for SMTCOMP2020": "yices_262"

            }

            if solver_name in solver_name_dict:
                solver_name = solver_name_dict[solver_name]
                

            base_name = f"{dataset_class}_{solver_name}_{timestamp}"

            print("\nStarting to save dataset")

            if save_format in ["pickle", "both"]:
                pickle_path = os.path.join(save_path, f"{base_name}.pkl")
                with open(pickle_path, 'wb') as f:
                    pickle.dump(dataset, f)
                print(f"having been saved in pickle format: {pickle_path}")
            
                # 保存数据集元信息
                metadata = {
                    'dataset_size': len(dataset),
                    'solver': solver_name,
                    'timestamp': timestamp,
                    'data_type': 'PyTorch Geometric Data objects',
                    'features_info': 'SMT features extracted from benchmark files'
                }
                
                metadata_path = os.path.join(save_path, f"{base_name}_metadata.pkl")
                with open(metadata_path, 'wb') as f:
                    pickle.dump(metadata, f)
                print(f"已保存元数据: {metadata_path}")

            if save_format in ["torch", "both"]:
                torch_path = os.path.join(save_path, f"{base_name}.pt")
                torch.save(dataset, torch_path)
                print(f"已保存为torch格式: {torch_path}")
        except Exception as e:
            print(f"保存数据集时出错: {e}")
            raise


if __name__ == "__main__":
    ''''
    这一部分的功能就是对文件进行预处理
    最后是把一个列表序列化 列表里的每个元素都是一个 torch_geometric.data.Data
    每个data对象的属性包括：
    x：节点特征矩阵 类型为 FloatTensor
    edge_index: 边索引，形状为2 num_edges 类型为LongTensor
    dge_attr（可选）：边的特征 类型为 FloatTensor
    y（可选）：标签，这里是对数归一化后的求解时间，类型为 FloatTensor
    solver（可选）：字符串，表示使用的求解器
    result（可选）：字符串或其它类型，表示求解结果（代码里暂时是 None）

    举例说明：
    features = {
        'node_features': [
            [0.1, 0.2, 0.3],   # 节点1的特征
            [0.4, 0.5, 0.6],   # 节点2的特征
            [0.7, 0.8, 0.9],   # 节点3的特征
        ],
        'edge_index': [
            [0, 1, 2],         # 边的起点
            [1, 2, 0]          # 边的终点
        ],
        'edge_attr': [
            [1.0],             # 边1的特征
            [2.0],             # 边2的特征
            [3.0],             # 边3的特征
        ]
    }

    '''

    ## 下面这些是处理数据需要修改的
    csv_file = "/root/autodl-tmp/project_gnn_original_data/smt_lib/QF_LIA/csv/new_qf_lia.csv"
    save_path = "/root/autodl-tmp/project_gnn_original_data/smt_lib/QF_LIA/bin"
    data_class = "QF_LIA"
    ## 还有上面这个_save_dataset的 如果有新的求解器名字 solver_name_dict也要更新
    ## pd.read_csv也要 formal 和 smt-lib修改 信息修改

    solver_names = [
                    # "Boolector-wrapped-sq", 
                    # "CVC4-2019-06-03-d350fe1-wrapped-sq",
                    # "UltimateEliminator+MathSAT-5.5.4-wrapped-sq",  
                    # "UltimateEliminator+Yices-2.6.1-wrapped-sq",
                    # "vampire-4.4-smtcomp-wrapped-sq",
                    # # "Poolector-wrapped-sq", 
                    # # "Q3B-wrapped-sq", 
                     
                    # "z3-4.8.4-d6df51951f4c-wrapped-sq",
                    # "COLIBRI 20.5.25",
                    "CVC4-sq-final",
                    "MathSAT5",
                    "veriT",
                    "smtinterpol-2.5-671-g6d0a7c6e",
                    "z3-4.8.8",
                    "Yices 2.6.2 for SMTCOMP2020"
                    ] 
    

    for solver_name in solver_names:
        create_smt_dataset_from_csv(csv_file, solver_name, save_path, data_class, "torch")




