from Models.BaseModel import BaseModel
from Datasets.BaseDataset import BaseDataset
from typing import Dict, Tuple, List
import numpy as np
import torch
from collections import defaultdict
import pandas as pd
from Datasets.DatasetUtils import *
import os
from Datasets.SMTGraghDataset import SMTGraghDataset
import json



class ModelEvaluator:
    ''' model evaluation'''
    def __init__(self, model: BaseModel, dataset: BaseDataset):
        self.model = model

        self.dataset = dataset
        self.model.eval()

        # 获取实际的模型对象来访问属性
        self.actual_model = model.module if hasattr(model, 'module') else model

    def _denormalize_time(self, normalized_time: float) -> float:
        """将归一化时间转换回实际时间"""
        return np.exp(normalized_time) - 1e-8


    

    def evaluate_all_metrics(self, predictions_file: str = None) -> Dict:
        predictions, targets, benchmark_paths, solvers = self._collect_predictions(predictions_file)
        benchmark_groups = self._group_by_benchmark(predictions, targets, benchmark_paths, solvers)

        results = {}
        
        # MAE
        results['MAE'] = self._compute_mae(predictions, targets)

        # MSE
        results["MSE"] = self._compute_mse(predictions, targets)

        # ACC
        results['ACC'] = self._compute_solver_selection_accuracy(benchmark_groups)

        # Time 
        results['Actual_Time'] = self._compute_actual_time(benchmark_groups)

        # **新增：虚拟最佳求解器总时间**
        results['VBS_Time'] = self._compute_vbs_time(benchmark_groups)

        # total cost
        results['Average_Cost'] = self._compute_average_cost(benchmark_groups)

        # 分段分析 - 按时间范围分组
        time_segment_results = self._compute_time_segment_analysis(benchmark_groups)
        results.update(time_segment_results)


        results['Top_3_ACC'] = self._compute_topk_accuracy(benchmark_groups, 3)
        results['Top_5_ACC'] = self._compute_topk_accuracy(benchmark_groups, 5)

        results['Kendall_Tau'] =  self._compute_kendall_tau(benchmark_groups)

        results["Relative_Error"] = self._compute_relative_error(predictions, targets)

        return results

    def _compute_vbs_time(self, benchmark_groups: Dict) -> float:
        """计算虚拟最佳求解器的总时间"""
        total_vbs_time = 0

        for benchmark_path, data_list in benchmark_groups.items():
            # 对每个benchmark，找到实际最佳时间（最小值）
            best_time = min(d['target'] for d in data_list)
            total_vbs_time += best_time

        return total_vbs_time


    def _collect_predictions(self, predictions_file: str = None) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """收集所有测试结果"""
        # 如果已有预测文件，直接加载
        if predictions_file and os.path.exists(predictions_file):
            print(f"从文件加载预测结果: {predictions_file}")
            with open(predictions_file, 'r') as f:
                data = json.load(f)
            return (
                np.array(data['predictions']),
                np.array(data['targets']),
                data['benchmark_paths'],
                data['solvers']
            )
        
        # 否则进行推理
        all_predictions = []
        all_targets = []
        all_benchmark_paths = []
        all_solvers = []

        with torch.no_grad():
            for data_dict in self.dataset:
                # 获取预测结果
                _, output_dict = self.actual_model.evalStep(data_dict)
                predictions = output_dict['pred'].cpu().numpy()
                
                # 获取真实标签和原始数据
                targets = data_dict['y'].cpu().numpy()

                # **新增：将归一化时间转换回实际时间**
                predictions = np.array([self._denormalize_time(p) for p in predictions])
                targets = np.array([self._denormalize_time(t) for t in targets])

                # 修正：使用实际的batch大小
                batch_size = len(targets)  # 或者 len(predictions)
                # 添加调试信息
                # print(f"Debug: predictions.shape={predictions.shape}, targets.shape={targets.shape}, batch_size={batch_size}")

                for i in range(batch_size):
                    all_predictions.append(predictions[i])
                    all_targets.append(targets[i])
                    all_benchmark_paths.append(data_dict['benchmark_path'][i])
                    all_solvers.append(data_dict['solver'][i])

        # 保存预测结果
        if predictions_file:
            prediction_data = {
                'predictions': [float(x) for x in all_predictions],  # 转换为可JSON序列化的格式
                'targets': [float(x) for x in all_targets],
                'benchmark_paths': all_benchmark_paths,
                'solvers': all_solvers
            }
            os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
            with open(predictions_file, 'w') as f:
                json.dump(prediction_data, f, indent=2)
            print(f"预测结果已保存到: {predictions_file}")

        return (
            np.array(all_predictions),
            np.array(all_targets),
            all_benchmark_paths,
            all_solvers
        )

    def _group_by_benchmark(self, predictions: np.ndarray, targets: np.ndarray, benchmark_paths: Dict, solvers: Dict) -> Dict:
        """根据benchmark分组"""

        groups = defaultdict(list)

        for pred, target, benchmark, solver in zip(predictions, targets, benchmark_paths, solvers):
            groups[benchmark].append({
                'prediction': pred,
                'target': target,
                'solver': solver
            })
        
        return dict(groups)

    def _compute_mae(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算平均绝对误差"""
        return np.mean(np.abs(predictions - targets))
    
    def _compute_mse(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算方均误差"""

        return np.mean((predictions - targets) ** 2)
    
    def _compute_solver_selection_accuracy(self, benchmark_groups: Dict) -> float:
        """ 计算的准确率"""
        correct_selections = 0
        total_benchmarks = len(benchmark_groups)

        for benchmark_path, data_list in benchmark_groups.items():
            min_pred_idx = np.argmin([d['prediction'] for d in data_list])
            pred_best_solver = data_list[min_pred_idx]['solver']

            min_actual_idx = np.argmin([d['target'] for d in data_list])    
            actual_best_solver = data_list[min_actual_idx]['solver']

            if pred_best_solver == actual_best_solver:
                correct_selections += 1

        return correct_selections / total_benchmarks if total_benchmarks > 0 else 0.0


    def _compute_actual_time(self, benchmark_groups: Dict) -> float:

        total_time = 0

        for benchmark_path, data_list in benchmark_groups.items():
            pred_best_idx = np.argmin([d['prediction'] for d in data_list])
            actual_time = data_list[pred_best_idx]['target']
            total_time += actual_time

        return total_time
    

    def _compute_average_cost(self, benchmark_groups: Dict) -> float:
        """计算相比虚拟最佳求解器的消耗时间"""

        total_cost = 0

        for benchmark_path, data_list in benchmark_groups.items():
            pred_best_idx = np.argmin([d['prediction'] for d in data_list])
            pred_best_time = data_list[pred_best_idx]['target']
            actual_best_time = min([d['target'] for d in data_list])

            total_cost += pred_best_time - actual_best_time

        return total_cost / len(benchmark_groups) if len(benchmark_groups) > 0 else 0.0
    
    def _compute_time_segment_analysis(self, benchmark_groups: Dict) -> Dict:
        """按时间范围分段分析模型性能"""
        
        # 收集所有benchmark的最优时间
        all_best_times = []
        for benchmark_path, data_list in benchmark_groups.items():
            best_time = min(d['target'] for d in data_list)
            all_best_times.append(best_time)
        
        all_best_times = np.array(all_best_times)
        
        # 按时间范围分成4段
        min_time = np.min(all_best_times)
        max_time = np.max(all_best_times)
        
        # 使用对数刻度分段，因为求解时间通常呈指数分布
        log_min = np.log10(max(min_time, 1e-6))  # 避免log(0)
        log_max = np.log10(max_time)
        
        segment_boundaries = np.logspace(log_min, log_max, 5)  # 5个边界点，4个段
        
        segment_results = {}
        
        for i in range(4):
            segment_name = f"Segment_{i+1}"
            lower_bound = segment_boundaries[i]
            upper_bound = segment_boundaries[i+1]
            
            # 筛选该时间段的benchmark
            segment_benchmarks = {}
            segment_count = 0
            
            for benchmark_path, data_list in benchmark_groups.items():
                best_time = min(d['target'] for d in data_list)
                if lower_bound <= best_time < upper_bound or (i == 3 and best_time <= upper_bound):
                    segment_benchmarks[benchmark_path] = data_list
                    segment_count += 1
            
            if segment_count == 0:
                segment_results[f"{segment_name}_Count"] = 0
                segment_results[f"{segment_name}_ACC"] = 0.0
                segment_results[f"{segment_name}_Average_Cost"] = 0.0
                segment_results[f"{segment_name}_Time_Range"] = f"[{lower_bound:.3f}, {upper_bound:.3f}]"
                continue
            
            # 计算该段的指标
            segment_acc = self._compute_solver_selection_accuracy(segment_benchmarks)
            segment_cost = self._compute_average_cost(segment_benchmarks)
            
            segment_results[f"{segment_name}_Count"] = segment_count
            segment_results[f"{segment_name}_ACC"] = segment_acc
            segment_results[f"{segment_name}_Average_Cost"] = segment_cost
            segment_results[f"{segment_name}_Time_Range"] = f"[{lower_bound:.3f}, {upper_bound:.3f}]"
        
        return segment_results



    def _compute_topk_accuracy(self, benchmark_groups: Dict, k: int) -> float:
        correct_selections = 0
        total_benchmarks = len(benchmark_groups)

        for benchmark_path, data_list in benchmark_groups.items():
            sorted_by_pred =  sorted(enumerate(data_list), key=lambda x:x[1]['prediction'])
            top_k_idx = [x[0] for x in sorted_by_pred[:k]]

            acutal_best_idx = np.argmin([d['target'] for d in data_list])

            if acutal_best_idx in top_k_idx:
                correct_selections += 1

        return correct_selections / total_benchmarks if total_benchmarks > 0 else 0.0
    

    def _compute_kendall_tau(self, benchmark_groups: Dict) -> float:
        from scipy.stats import kendalltau

        tau_values = []

        for benchmark_path, data_list in benchmark_groups.items():
            if len(data_list) < 2:
                continue

            predictions =  [d['prediction'] for d in data_list]
            targets = [d['target'] for d in data_list]

            tau, _ = kendalltau(predictions, targets)

            if not np.isnan(tau):
                tau_values.append(tau)

        return np.mean(tau_values) if tau_values else 0.0
    
    def _compute_relative_error(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        mask = targets != 0

        if sum(mask) == 0:
            return 0.0
        
        relative_errors = np.abs((predictions[mask] - targets[mask])) / targets[mask]

        return np.mean(relative_errors)
    


    
    def generate_detailed_report(self, save_path: str = None, predictions_file: str = None) -> pd.DataFrame:
        results = self.evaluate_all_metrics(predictions_file)

        report_data = []
        for metric, value in results.items():
            report_data.append({
                'Metric': metric,
                'Value': value,
                'Description': self._get_metric_descriptions(metric)
            })

        report_df = pd.DataFrame(report_data)

        if save_path:
            report_df.to_csv(save_path, index=False)
            print(f"评估报告已保存到: {save_path}")

        return report_df


    def _get_metric_descriptions(self, metric: str) -> str:
        """获取指标描述"""
        descriptions = {
            'MAE': '平均绝对误差 - 所有预测的平均绝对偏差（实际时间，秒）',
            'MSE': '均方误差 - 所有预测的均方偏差（实际时间，秒²）',
            'ACC': '求解器选择准确率 - 正确选择最佳求解器的比例',
            'Actual_Time': '实际总时间 - 使用预测最佳求解器的总时间（秒）',
            'VBS_Time': '虚拟最佳求解器总时间 - 理论最优情况下的总时间（秒）',  # **新增**
            'Average_Cost': '平均额外成本 - 相比虚拟最佳求解器的平均额外时间（秒）',
            'Top_3_ACC': 'Top-3准确率 - 最佳求解器在预测前3名中的比例',
            'Top_5_ACC': 'Top-5准确率 - 最佳求解器在预测前5名中的比例',
            'Kendall_Tau': 'Kendall相关系数 - 预测排序与真实排序的相关性',
            'Relative_Error': '相对误差 - 预测误差相对于真实值的比例'
        }
        
        # 添加分段分析的描述
        for i in range(1, 5):
            descriptions[f'Segment_{i}_Count'] = f'第{i}段benchmark数量'
            descriptions[f'Segment_{i}_ACC'] = f'第{i}段求解器选择准确率'
            descriptions[f'Segment_{i}_Average_Cost'] = f'第{i}段平均额外成本（秒）'
            descriptions[f'Segment_{i}_Time_Range'] = f'第{i}段时间范围（秒）'
        
        return descriptions.get(metric, '未知指标')



    

def evaluate_model(model_path: str, data_dir: str, output_dir: str):
    """评估训练好的模型"""
    
    model_path = os.path.join(model_path, 'best.pth')
    # 加载模型
    from Models.Simple.SimpleGNN import SimpleGNN
    model = SimpleGNN(node_features_dim=20, num_solvers=6, optimizer_cls=torch.optim.Adam,
                                       optimizer_args={"lr": 1e-5})
    model.load_state_dict(torch.load(model_path))
    
    # 在测试时也使用DataParallel
    if torch.cuda.device_count() > 1:
        print(f"测试时使用{torch.cuda.device_count()}个GPU")
        model = torch.nn.DataParallel(model)
    
    model.to(DEVICE)
    model.eval()
    
    # 加载测试数据
    from DynamicConfig import DynamicConfig
    dataset_cfg = DynamicConfig(SMTGraghDataset,
                                data_dir=data_dir,
                                batch_size=16,  # 增加批次大小
                                drop_last=False,
                                shuffle=False,
                                max_files_in_memory=5,  # 增加文件缓存
                                preload_files=True)
    dataset_cfg.set_name = "all"
    
    
    test_dataset = dataset_cfg.build()

    # 创建评估器
    evaluator = ModelEvaluator(model, test_dataset)
    
    # 生成评估报告
    os.makedirs(output_dir, exist_ok=True)
    # predictions_file = os.path.join(output_dir, "predictions.json")
    predictions_file = None
    report_path = os.path.join(output_dir, "evaluation_report.csv")
    report_df = evaluator.generate_detailed_report(report_path, predictions_file)
    
    # 打印结果
    print("\n=== 模型评估结果 ===")
    for _, row in report_df.iterrows():
        print(f"{row['Metric']}: {row['Value']}")
        print(f"  描述: {row['Description']}\n")
    
    return evaluator, report_df


if __name__ == "__main__":
    # 使用示例
    model_path = "/root/project_gnn/Runs/SMTGraghDataset/SimpleGNN/250904_124335"
    data_dir = [
        '/root/autodl-tmp/project_gnn_original_data/formal_verification_data/CC/bin',
        '/root/autodl-tmp/project_gnn_original_data/formal_verification_data/FPV/bin',
        '/root/autodl-tmp/project_gnn_original_data/formal_verification_data/UNR/bin'
    ]
    output_dir = "/root/project_gnn/Runs/SMTGraghDataset/SimpleGNN/250904_124335"
    
    evaluator, report = evaluate_model(model_path, data_dir, output_dir)