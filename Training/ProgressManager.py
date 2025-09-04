from typing import List
from rich.console import Console
from rich.live import Live
from rich.table import Table, Column
import time
import threading
from threading import Thread


class ProgressManager:
    def __init__(self,
                 items_per_epoch: int,
                 epochs: int,
                 show_recent: int,
                 refresh_interval: int = 1,
                 custom_fields: List[str] = None):
        '''
        参数说明
        items_per_epoch：每轮（epoch）有多少个步骤（比如每个batch算一个步骤）
        epochs：总轮次（训练的总迭代轮数）
        show_recent：表格中显示最近多少轮的进度
        refresh_interval：界面刷新间隔（秒）
        custom_fields：自定义字段（比如要显示"loss"、"accuracy"等指标）
        '''
        
        
        # 保存基础配置
        self.epochs = epochs  # 总轮数
        self.steps_per_epoch = items_per_epoch  # 每轮的步骤数
        self.total_steps = epochs * items_per_epoch  # 总步骤数（所有轮次的步骤总和）
        self.display_recent = show_recent  # 显示最近的轮次数量
        self.refresh_interval = refresh_interval  # 刷新间隔
        self.custom_fields = [] if custom_fields is None else custom_fields  # 自定义字段列表

        # 计算表格总宽度（根据列数动态调整）
        self.total_width = (9 + 9 + 15 + 10 + 10) + 12 * len(self.custom_fields)


        # 初始化进度跟踪变量
        self.overall_progress = 0  # 整体完成的步骤数
        self.start_time = time.time()  # 开始时间（用于计算总耗时）
        self.console = Console(width=self.total_width + 6 + len(self.custom_fields))  # 终端输出对象
        self.live = None  # 用于实时更新的对象（后面会初始化）

        self.current_epoch = 1  # 当前轮次（从1开始）
        self.current_step = 1  # 当前步骤（从1开始）

        # 初始化每轮的进度数据（memory是一个列表，每个元素对应一轮的信息）
        self.memory = [
            ({"epoch": epoch, "completed": 0, "t_start": 0.0, "t_end": 0.0} | 
            dict(zip(self.custom_fields, [0]*len(custom_fields))))
            for epoch in range(1, epochs + 1)
        ]
        # 解释：每个epoch的信息包括：
        # - "epoch"：轮次号
        # - "completed"：已完成的步骤数
        # - "t_start"：开始时间（时间戳）
        # - "t_end"：结束时间（时间戳，未完成时为0）
        # - 自定义字段（如loss，初始值为0）


        # 整体进度行的样式模板（用rich的颜色语法美化）
        self.overall_row_forms = [
            "[#00aaff]Overall[#00aaff]",    # 整体进度行的标题
            "[#00aaff]{}%[/#00aaff]",       # 整体进度百分比
            "[#00aaff]{}/{}[/#00aaff]",     # 整体完成步骤/总步骤
            "[#00aaff]{}[/#00aaff]",        # 总耗时
            "[#00aaff]{}[/#00aaff]",        # 总剩余时间
        ]

    def update(self, current_epoch: int, current_step: int, **custom_values):
        # 第一次调用时，启动实时显示线程
        if not hasattr(self, "live_thread"):
            self.console.print("[bold green]Starting Training...[/bold green]")  # 打印开始信息
            self.live = Live(self.render_progress_table(1), refresh_per_second=1, console=self.console)  # 初始化实时更新对象
            self.start_time = time.time()  # 记录开始时间
            self.live_thread = Thread(target=self.live_update)  # 创建线程，目标函数是live_update
            self.live_thread.start()  # 启动线程

        self.overall_progress += 1

        # 更新当前轮次的进度数据
        self.memory[current_epoch]["completed"] = current_step + 1  # 记录当前轮次已完成的步骤（+1是因为步骤从0开始计数）
        # 更新自定义字段的值（比如传入loss=0.5，就更新memory中对应epoch的loss字段）
        for k in self.custom_fields:
            self.memory[current_epoch][k] = custom_values[k]

        # 更新当前轮次和步骤（为下一次更新做准备）
        self.current_epoch = current_epoch + 1
        self.current_step = current_step + 1

        # 记录轮次的开始和结束时间
        if self.memory[current_epoch]["t_start"] == 0:
            # 如果是新轮次，记录开始时间
            self.memory[current_epoch]["t_start"] = time.time()
            if current_epoch >= 1:
                # 上一轮结束，记录结束时间
                self.memory[current_epoch - 1]["t_end"] = time.time()

    def format_time(self, seconds: float) -> str:
        """将秒数转换为 hh:mm:ss 格式（比如3661秒 → 01:01:01）"""
        hrs, rem = divmod(int(seconds), 3600)  # 小时 = 总秒数 ÷ 3600，余数是剩下的秒数
        mins, secs = divmod(rem, 60)  # 分钟 = 剩余秒数 ÷ 60，余数是秒数
        return f"{hrs:02}:{mins:02}:{secs:02}"  # 补0成两位数（如1→01）
    
    
    def render_progress_table(self, current_epoch: int) -> Table:
        """创建并返回一个进度表格（用rich.Table）"""
        # 初始化表格：显示表头，表头样式为粗体洋红色，设置最小宽度
        table = Table(show_header=True, header_style="bold magenta", min_width=self.total_width)
        # 添加表格列（固定列 + 自定义字段列）
        table.add_column("Epoch", width=9)  # 轮次号
        table.add_column("Percent", width=9)  # 进度百分比
        table.add_column("Progress", width=15)  # 完成步骤/总步骤
        table.add_column("Elapsed", width=10)  # 已耗时
        table.add_column("Remain", width=10)  # 剩余时间
        for k in self.custom_fields:
            table.add_column(k, width=12)  # 自定义字段列（如loss）

        # 计算总耗时和总剩余时间
        elapsed_time_total = time.time() - self.start_time  # 总耗时 = 当前时间 - 开始时间
        # 剩余时间 = （总步骤 - 已完成步骤） × （平均每个步骤耗时）
        remaining_time_total = (self.total_steps - self.overall_progress) * (
                elapsed_time_total / self.overall_progress) if self.overall_progress > 0 else 0

        # 添加“整体进度行”
        overall_percentage = int(self.overall_progress / self.total_steps * 100)  # 整体进度百分比

        table.add_row(
            self.overall_row_forms[0],  # 标题“Overall”
            self.overall_row_forms[1].format(overall_percentage),  # 百分比
            self.overall_row_forms[2].format(self.overall_progress, self.total_steps),  # 完成步骤/总步骤
            self.overall_row_forms[3].format(self.format_time(elapsed_time_total)),  # 总耗时
            self.overall_row_forms[4].format(self.format_time(remaining_time_total)),  # 总剩余时间
            ""  # 自定义字段列留空（整体行不需要）
        )

        # 添加“最近轮次的进度行”（根据show_recent配置显示）
        # 遍历范围：从“当前轮次 - 要显示的最近轮次”到“当前轮次”
        for i in range(max(0, current_epoch - self.display_recent), current_epoch):
            epoch_data = self.memory[i]  # 获取第i轮的进度数据
            # 计算当前轮次的进度百分比
            epoch_percentage = int(epoch_data["completed"] / self.steps_per_epoch * 100)

            # 判断轮次是否已完成，计算耗时和剩余时间
            if epoch_data["t_end"] == 0:
                # 未完成：耗时 = 当前时间 - 轮次开始时间；剩余时间 = 剩余步骤 × 平均步骤耗时
                elapsed_time_epoch = time.time() - epoch_data["t_start"]
                remaining_time_epoch = (self.steps_per_epoch - epoch_data["completed"]) * (
                            elapsed_time_epoch / epoch_data["completed"]) if epoch_data["completed"] > 0 else 0
                color = "#ffff00"  # 未完成用黄色
            else:
                # 已完成：耗时 = 轮次结束时间 - 开始时间；剩余时间为0
                elapsed_time_epoch = epoch_data["t_end"] - epoch_data["t_start"]
                remaining_time_epoch = 0
                color = "green"  # 已完成用绿色

            # 向表格添加当前轮次的行
            table.add_row(
                f"[{color}]{epoch_data['epoch']}[/{color}]",  # 轮次号（带颜色）
                f"[{color}]{epoch_percentage}%[/{color}]",  # 百分比（带颜色）
                f"[{color}]{epoch_data['completed']}/{self.steps_per_epoch}[/{color}]",  # 完成步骤/总步骤
                f"[{color}]{self.format_time(elapsed_time_epoch)}[/{color}]",  # 已耗时
                f"[{color}]{self.format_time(remaining_time_epoch)}[/{color}]",  # 剩余时间
                # 自定义字段值（科学计数法保留3位小数）
                *[f"{epoch_data[k]:.3e}" for k in self.custom_fields]
            )

        return table
    
    def live_update(self):
        with self.live:  # 启动实时更新上下文
            # 只要没完成所有步骤，就不断刷新表格
            while self.overall_progress != self.total_steps:
                self.live.update(self.render_progress_table(self.current_epoch))  # 更新表格内容
                time.sleep(self.refresh_interval)  # 按间隔时间休眠

    def close(self):
        """结束后关闭实时显示"""
        if hasattr(self, "live_thread"):
            self.live.stop()  # 停止实时更新
            self.live_thread.join()  # 等待线程结束
            del self.live_thread  # 删除线程对象
            del self.live  # 删除live对象
            self.console.print("[bold green]Training Completed![/bold green]")  # 打印完成信息

