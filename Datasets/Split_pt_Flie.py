import torch
import os
from math import ceil

def split_large_pt_file(input_file_path, output_dir, files_per_split=2500):
    """
    分割大的.pt文件
    
    Args:
        input_file_path: 输入文件路径
        output_dir: 输出目录
        files_per_split: 每个分割文件包含的样本数
    """
    print(f"开始分割文件: {input_file_path}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载原始数据
    print("正在加载原始数据...")
    try:
        original_data = torch.load(input_file_path, weights_only=False)
    except Exception as e:
        print(f"加载失败，尝试使用weights_only=False: {e}")
        original_data = torch.load(input_file_path, weights_only=False)
    
    total_samples = len(original_data)
    print(f"总样本数: {total_samples}")
    
    # 计算需要分割的文件数
    num_splits = ceil(total_samples / files_per_split)
    print(f"将分割为 {num_splits} 个文件，每个文件约 {files_per_split} 个样本")
    
    # 获取原始文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(input_file_path))[0]
    
    # 分割并保存
    for i in range(num_splits):
        start_idx = i * files_per_split
        end_idx = min((i + 1) * files_per_split, total_samples)
        
        # 提取当前分割的数据
        split_data = original_data[start_idx:end_idx]
        
        # 构造输出文件名
        output_file = os.path.join(output_dir, f"{base_name}_part_{i+1:02d}.pt")
        
        # 保存分割文件
        print(f"保存分割文件 {i+1}/{num_splits}: {output_file}")
        print(f"  样本范围: {start_idx} - {end_idx-1} (共 {len(split_data)} 个样本)")
        
        torch.save(split_data, output_file)
        
        # 释放内存
        del split_data
    
    print("分割完成！")
    
    # 清理原始数据
    del original_data
    
    return num_splits

# 使用示例
if __name__ == "__main__":
    input_file = "/root/autodl-tmp/project_gnn_original_data/FPV/test/FPV_test.pt"
    output_directory = "/root/autodl-tmp/project_gnn_original_data/FPV/test_split/"


    
    # 每个文件约2000个样本（约5G大小）
    split_large_pt_file(input_file, output_directory, files_per_split=2000)
