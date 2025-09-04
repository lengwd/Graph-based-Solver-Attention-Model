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
import time
import signal
from contextlib import contextmanager
from  .DatasetUtils import * 

import argparse




def detect_csv_format(csv_file):
    """检测CSV文件格式并返回列名映射"""
    # 读取第一行来检测格式
    df_sample = pd.read_csv(csv_file, nrows=1)
    columns = df_sample.columns.tolist()
    
    if len(columns) == 4 and 'sat/unsat' in columns:
        # 表格1格式: benchmark,solver,sat/unsat,score
        print("检测到表格格式1: benchmark,solver,sat/unsat,score")
        return {
            'benchmark': 'benchmark',
            'solver': 'solver', 
            'solve_time': 'score',
            'result': 'sat/unsat'
        }
    elif len(columns) == 3 and 'score' in columns:
        # 表格2格式: benchmark,solver,score
        print("检测到表格格式2: benchmark,solver,score")
        return {
            'benchmark': 'benchmark',
            'solver': 'solver',
            'solve_time': 'score',
            'result': None
        }
    else:
        # 尝试自动检测
        print(f"未知格式，列名: {columns}")
        # 假设最后一列是score/solve_time
        return {
            'benchmark': columns[0],
            'solver': columns[1],
            'solve_time': columns[-1],
            'result': columns[2] if len(columns) > 3 else None
        }



class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    """超时上下文管理器"""
    def signal_handler(signum, frame):
        raise TimeoutError(f"操作超时 ({seconds}秒)")
    
    # 设置信号处理器
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # 恢复原来的信号处理器
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def smt_gnn_parse_to_pyg_data(features, solve_time=None, solver=None, result=None, benchmark_path=None, debug=False):
    """translate the format extract by gnn_parse.py to pyg data format"""

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

        if debug:
            print(f"原始时间: {solve_time}")
            print(f"对数变换后: {normalized_time}")
            print(f"张量y: {data.y}")
            print(f"y的实际值: {data.y.item()}")

    if solver is not None:
        data.solver = solver
    
    if result is not None:
        data.result = result

    if benchmark_path is not None:
        data.benchmark_path = benchmark_path

    return data


def create_smt_dataset_from_csv_optimized(csv_file, target_solvers=None, save_path=None, dataset_class=None, save_format='pickle', timeout_seconds=60):
    """
    优化版本：每个文件只解析一次，然后为所有目标求解器创建数据
    
    Args:
        csv_file: CSV文件路径
        target_solvers: 目标求解器列表，如果为None则处理所有求解器
        save_path: 保存路径，如果为None则不保存
        dataset_class: 数据集类别名称
        save_format: 保存格式，支持'pickle', 'torch', 'both'
    
    Returns:
        datasets: 字典，键为求解器名称，值为对应的数据集列表
    """
    # 检测CSV格式
    column_mapping = detect_csv_format(csv_file)
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 重命名列以统一格式
    rename_dict = {}
    for standard_name, original_name in column_mapping.items():
        if original_name and original_name in df.columns:
            rename_dict[original_name] = standard_name
    
    df = df.rename(columns=rename_dict)
    
    # 确保必要的列存在
    required_columns = ['benchmark', 'solver', 'solve_time']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要的列: {missing_columns}")
    
    # 如果没有指定目标求解器，则使用所有求解器
    if target_solvers is None:
        target_solvers = df['solver'].unique().tolist()
    
    # 过滤出目标求解器的数据
    target_data = df[df['solver'].isin(target_solvers)]
    
    if target_data.empty:
        print(f"警告：没有找到目标求解器的数据")
        return {}
    
    print(f"目标求解器: {target_solvers}")
    print(f"总数据条数: {len(target_data)}")
    
    # 按求解器统计
    for solver in target_solvers:
        solver_data = target_data[target_data['solver'] == solver]
        print(f"{solver} 数据条数: {len(solver_data)}")
        if len(solver_data) > 0:
            print(f"{solver} 平均求解时间: {solver_data['solve_time'].mean():.2f}")

    # 按benchmark分组
    benchmark_groups = list(target_data.groupby("benchmark"))
    
    # 初始化每个求解器的数据集
    datasets = {solver: [] for solver in target_solvers}
    successful_count = 0
    failed_count = 0
    timeout_count = 0  # 添加这行
    
    print(f"开始处理 {len(benchmark_groups)} 个benchmark文件...")

    for benchmark_path, group in tqdm(benchmark_groups, desc='处理文件', unit='文件'):

        
        try:
            # 使用超时上下文管理器，60秒超时
            with timeout(timeout_seconds):
             # 每个文件只解析一次
                parser = YosysSMTtoTree()
                parser.parse_smt_file(benchmark_path)
                features = parser.extract_features()
                
                # 为该文件的所有求解器结果创建数据
                for solver in target_solvers:
                    solver_results = group[group['solver'] == solver]
                    
                    for _, row in solver_results.iterrows():
                        current_solver_name = row["solver"]
                        solve_result = None
                        solve_time = row['solve_time']

                        data = smt_gnn_parse_to_pyg_data(
                            features, 
                            solve_time, 
                            current_solver_name, 
                            solve_result,
                            benchmark_path=benchmark_path
                        )
                        datasets[solver].append(data)
                
                successful_count += 1

        except TimeoutError as e:
            timeout_count += 1
            tqdm.write(f"⏰ 跳过超时文件 {benchmark_path}: {e}")
            continue
        except Exception as e:
            failed_count += 1
            tqdm.write(f"❌ 处理文件 {benchmark_path} 时出错: {e}")
            continue

    print(f"\n处理完成")
    print(f"成功处理: {successful_count} 个文件")
    print(f"失败处理: {failed_count} 个文件")
    print(f"超时跳过: {timeout_count} 个文件")  # 新增
    
    # 打印每个求解器的最终数据集大小
    for solver in target_solvers:
        print(f"{solver} 最终数据集大小: {len(datasets[solver])}")

    # 保存数据集
    if save_path is not None:
        for solver, dataset in datasets.items():
            if len(dataset) > 0:  # 只保存非空数据集
                _save_dataset(dataset, save_path, dataset_class, save_format, solver)

    return datasets


def create_combined_dataset_from_csv(csv_file, target_solvers=None, save_path=None, dataset_class=None, save_format='pickle', timeout_seconds=60):
    """
    创建合并的数据集：所有求解器的数据放在一个文件中
    每个数据点包含所有求解器的求解时间信息
    
    Args:
        csv_file: CSV文件路径
        target_solvers: 目标求解器列表
        save_path: 保存路径
        dataset_class: 数据集类别名称
        save_format: 保存格式
    
    Returns:
        dataset: 合并后的数据集列表
    """
    
    # 检测CSV格式
    column_mapping = detect_csv_format(csv_file)
    
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 重命名列以统一格式
    rename_dict = {}
    for standard_name, original_name in column_mapping.items():
        if original_name and original_name in df.columns:
            rename_dict[original_name] = standard_name
    
    df = df.rename(columns=rename_dict)
    
    # 确保必要的列存在
    required_columns = ['benchmark', 'solver', 'solve_time']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"缺少必要的列: {missing_columns}")
    
    if target_solvers is None:
        target_solvers = df['solver'].unique().tolist()
    
    # 按benchmark分组
    benchmark_groups = list(df.groupby("benchmark"))
    
    dataset = []
    successful_count = 0
    failed_count = 0
    timeout_count = 0  # 添加这行
    
    print(f"开始处理 {len(benchmark_groups)} 个benchmark文件...")

    for benchmark_path, group in tqdm(benchmark_groups, desc='处理文件', unit='文件'):

        

        try:
            # 检查是否有目标求解器的数据
            available_solvers = set(group['solver'].unique()) & set(target_solvers)
            if not available_solvers:
                continue
                
            # 使用超时上下文管理器，60秒超时
            with timeout(timeout_seconds):
                # 每个文件只解析一次
                parser = YosysSMTtoTree()
                parser.parse_smt_file(benchmark_path)
                features = parser.extract_features()
                
                # 创建求解时间字典
                solve_times = {}
                for _, row in group.iterrows():
                    solver = row['solver']
                    if solver in target_solvers:
                        solve_times[solver] = row['solve_time']
                
                # 创建一个包含所有求解器信息的数据点
                x = torch.FloatTensor(features['node_features'])
                edge_index = torch.LongTensor(features['edge_index'])
                edge_attr = torch.FloatTensor(features["edge_attr"]) if len(features["edge_attr"]) > 0 else None

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                
                # 添加所有求解器的时间信息
                data.solve_times = solve_times
                data.benchmark_path = benchmark_path
                data.available_solvers = list(available_solvers)
                
                dataset.append(data)
                successful_count += 1

        except TimeoutError as e:
            timeout_count += 1
            tqdm.write(f"⏰ 跳过超时文件 {benchmark_path}: {e}")
            continue
        except Exception as e:
            failed_count += 1
            tqdm.write(f"❌ 处理文件 {benchmark_path} 时出错: {e}")
            continue

    print(f"\n处理完成")
    print(f"成功处理: {successful_count} 个文件")
    print(f"失败处理: {failed_count} 个文件")
    print(f"超时跳过: {timeout_count} 个文件")  # 新增
    print(f"最终数据集大小: {len(dataset)}")


    # 保存合并的数据集
    if save_path is not None:
        _save_combined_dataset(dataset, save_path, dataset_class, save_format, target_solvers)

    return dataset


def _save_dataset(dataset, save_path, dataset_class, save_format, solver_name):
    """保存单个求解器的数据集"""
    try:
        if save_path and os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
            "smtinterpol-2.5-671-g6d0a7c6e": "smtinterpol",
            "z3-4.8.8": "z3_488",
            "Yices 2.6.2 for SMTCOMP2020": "yices_262"
        }

        if solver_name in solver_name_dict:
            solver_name = solver_name_dict[solver_name]

        base_name = f"{dataset_class}_{solver_name}_{timestamp}"

        print(f"\n开始保存 {solver_name} 的数据集")

        if save_format in ["pickle", "both"]:
            pickle_path = os.path.join(save_path, f"{base_name}.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"已保存为pickle格式: {pickle_path}")
        
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


def _save_combined_dataset(dataset, save_path, dataset_class, save_format, solver_names):
    """保存合并的数据集"""
    try:
        if save_path and os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        base_name = f"{dataset_class}_combined_{timestamp}"

        print(f"\n开始保存合并数据集")

        if save_format in ["pickle", "both"]:
            pickle_path = os.path.join(save_path, f"{base_name}.pkl")
            with open(pickle_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"已保存为pickle格式: {pickle_path}")
        
            # 保存数据集元信息
            metadata = {
                'dataset_size': len(dataset),
                'solvers': solver_names,
                'timestamp': timestamp,
                'data_type': 'PyTorch Geometric Data objects with multi-solver info',
                'features_info': 'SMT features with solve times for multiple solvers'
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


    parser = argparse.ArgumentParser(description="SMT Dataset Preprocess")

    parser.add_argument(
        '--data_class', '-d_cls',
        type=str,
        choices=['CC', 'FPV', 'UNR', 'QF_LIA', 'BV', 'NRA', 'QF_BVFPLRA', 'QF_UFBV'],
        default='CC',
        help='data class (default: CC)'
    )

    parser.add_argument(
        '--save_format', '-f',
        type=str,
        choices=['pickle', 'torch', 'both'],
        default='torch',
        help='save format (default: torch)'
    )

    parser.add_argument(
        '--timeout', '-t',
        type=int,
        default=60,
        help='超时时间（秒） (默认: 60)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        choices=['optimized', 'combined', 'both'],
        default='optimized',
        help="process mode (default: optimized)"
    )

    args = parser.parse_args()

    data_class = args.data_class
    save_format = args.save_format
    timeout_seconds = args.timeout
    mode = args.mode

    print(f"数据集类别: {data_class}")
    print(f"保存格式: {save_format}")
    print(f"超时时间: {timeout_seconds}秒")
    print(f"处理模式: {mode}")
    

    

    


    csv_file = csv_file_dict[data_class]
    save_path = save_path_dict[data_class]
    

    
    

    # 根据模式执行相应的处理
    if mode in ['optimized', 'both']:
        print("=== 使用优化版本处理数据 ===")
        datasets = create_smt_dataset_from_csv_optimized(
            csv_file, 
            target_solvers=solver_names[data_class], 
            save_path=save_path, 
            dataset_class=data_class, 
            save_format=save_format,
            timeout_seconds=timeout_seconds  # 需要修改函数签名
        )

    if mode in ['combined', 'both']:
        print("\n=== 创建合并数据集 ===")
        combined_dataset = create_combined_dataset_from_csv(
            csv_file, 
            target_solvers=solver_names[data_class], 
            save_path=save_path, 
            dataset_class=data_class, 
            save_format=save_format,
            timeout_seconds=timeout_seconds  # 需要修改函数签名
        )