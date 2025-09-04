from z3 import *
import networkx as nx
import re
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt 


class YosysSMTtoTree:

    def __init__(self):
        # 核心数据结构
        self.tree = nx.DiGraph()  # 主要的有向图结构
        
        # SMT基本元素存储
        self.declarations = {}    # 声明的函数和排序
        self.functions = {}       # 定义的函数
        self.assertions = {}      # 断言语句
        
        # Yosys硬件相关元素
        self.registers = {}       # 寄存器
        self.wires = {}          # 线网
        self.inputs = {}         # 输入端口
        self.outputs = {}        # 输出端口
        
        # 节点管理
        self.node_counter = 0    # 节点计数器
        self.node_map = {}       # 节点映射


    def parse_smt_file(self, smt2_file_path):
        """解析Yosys生成的SMT文件并构建树结构"""

        with open(smt2_file_path, 'r') as f:
            content = f.read()

        # 分步解析不同的SMT结构
        self._parse_declarations(content)      # 解析声明
        self._parse_define_functions(content)  # 解析函数定义
        self._parse_assertions(content)        # 解析断言
        self._parse_yosys_annotations(content) # 解析Yosys注释


        # 构建主要的树结构
        self._build_main_tree()

        return self.tree


    def extract_features(self):
        """提取图特征用于GNN"""

        if len(self.tree.nodes()) == 0:
            return {"error": "Empty Tree"}
        
        node_features = []
        edge_index = []
        edge_attr = []

        for node in self.tree.nodes():
            node_data = self.tree.nodes[node]
            features = [
                1 if node_data.get('type') == 'compound' else 0,
                1 if node_data.get('type') == 'bitvector_constant' else 0,
                1 if node_data.get('type') == 'identifier' else 0,
                1 if node_data.get('type') == 'symbol' else 0,
                1 if node_data.get('type') == 'module' else 0,
                1 if node_data.get('is_function_root', False) else 0,
                1 if node_data.get('type') == 'function_reference' else 0,  # 函数引用特征
                1 if node_data.get('type') == 'function_call' else 0,  # 函数调用特征
                1 if (node_data.get('is_assertion_root', False) or node_data.get('name', '').startswith('assertion_')) else 0,   # 添加断言特征
                1 if node_data.get('hardware_type') == 'register' else 0,
                1 if node_data.get('hardware_type') == 'wire' else 0,
                1 if node_data.get('hardware_type') == 'input' else 0,
                1 if node_data.get('hardware_type') == 'output' else 0,
                node_data.get('arity', 0),
                node_data.get('bit_width', 0),
                node_data.get('width', 0),
                node_data.get('depth', 0),
                self.tree.degree(node),
                len(list(self.tree.predecessors(node))),
                len(list(self.tree.successors(node)))
            ]

            node_features.append(features)

        nodes_mapping = {node: idx for idx, node in enumerate(self.tree.nodes())}

        for edge in self.tree.edges():
            src, dst = edge
            edge_index.append([nodes_mapping[src], nodes_mapping[dst]])

            edge_data = self.tree.edges[edge]
            edge_attr.append(
                [
                    1 if edge_data.get('edge_type') == 'child' else 0,
                    1 if edge_data.get('edge_type') == 'contains' else 0,
                    1 if edge_data.get('edge_type') == 'asserts' else 0,
                    1 if edge_data.get('edge_type') == 'function_call' else 0,  # 添加函数调用边
                    1  # 默认权重
                ]
            )

        return {
            'node_features': np.array(node_features),
            'edge_index': np.array(edge_index).T if edge_index else np.array([]).reshape(2, 0),
            'edge_attr': np.array(edge_attr) if edge_attr else np.array([]).reshape(0, 5),
            'num_nodes': len(self.tree.nodes()),
            'tree': self.tree,
            'statistics': self._get_statistics()
        }


    def print_summary(self):
        """打印解析结果摘要"""
        
        print("=== SMT文件解析摘要 ===")
        print(f"声明数量: {len(self.declarations)}")
        print(f"函数定义数量: {len(self.functions)}")
        print(f"断言数量: {len(self.assertions)}")
        print(f"寄存器数量: {len(self.registers)}")
        print(f"线网数量: {len(self.wires)}")
        print(f"输入端口数量: {len(self.inputs)}")
        print(f"输出端口数量: {len(self.outputs)}")
        print(f"树节点总数: {len(self.tree.nodes())}")
        print(f"树边总数: {len(self.tree.edges())}")
        
        # 显示一些示例函数
        print("\n=== 示例函数 ===")
        for i, (func_name, func_info) in enumerate(list(self.functions.items())[:5]):
            print(f"{i+1}. {func_name}: {func_info['return_type']}")
            body_preview = func_info['body'][:100] + '...' if len(func_info['body']) > 100 else func_info['body']
            print(f"   Body: {body_preview}")
        
        if len(self.functions) > 5:
            print(f"   ... 还有 {len(self.functions) - 5} 个函数")



    def visualize_tree(self, max_nodes=100, save_path="/root/workspace/practice/z3_api/testcase.png"):
        """可视化树结构，采用分层布局从上到下显示"""
        
        if len(self.tree.nodes()) > max_nodes:
            print(f"树太大，总共 {len(self.tree.nodes())} 个节点，只显示前 {max_nodes} 个节点。")
            node_to_show = list(self.tree.nodes())[:max_nodes]
            subgraph = self.tree.subgraph(node_to_show)
        else:
            subgraph = self.tree

        plt.figure(figsize=(20, 12))

        # 使用分层布局
        try:
            # 首先尝试使用graphviz的dot布局（如果可用）
            try:
                pos = nx.nx_agraph.graphviz_layout(subgraph, prog='dot')
            except:
                # 如果graphviz不可用，使用自定义的分层布局
                pos = self._create_hierarchical_layout(subgraph)
        except:
            # 后备方案：使用spring布局
            pos = nx.spring_layout(subgraph, k=3, iterations=50)

        # 节点颜色映射
        node_colors = []
        node_sizes = []
        
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            
            # 根据节点类型设置颜色
            if node_data.get('type') == 'module':
                node_colors.append('#FF4444')  # 红色 - 模块根节点
                node_sizes.append(800)
            elif node_data.get('is_function_root'):
                node_colors.append('#4444FF')  # 蓝色 - 函数根节点
                node_sizes.append(600)
            elif node_data.get('type') == 'compound':
                if node_data.get('operator_type') == 'conditional':
                    node_colors.append('#FF8800')  # 橙色 - 条件操作
                elif node_data.get('operator_type') == 'logical':
                    node_colors.append('#88FF00')  # 绿色 - 逻辑操作
                elif node_data.get('operator_type') == 'comparison':
                    node_colors.append('#00FF88')  # 青绿色 - 比较操作
                elif node_data.get('operator_type') == 'bitvector_arithmetic':
                    node_colors.append('#8800FF')  # 紫色 - 位向量算术
                else:
                    node_colors.append('#00AA00')  # 深绿色 - 其他复合操作
                node_sizes.append(400)
            elif node_data.get('hardware_type') == 'register':
                node_colors.append('#FFAA00')  # 橙黄色 - 寄存器
                node_sizes.append(350)
            elif node_data.get('hardware_type') == 'wire':
                node_colors.append('#FFFF00')  # 黄色 - 线网
                node_sizes.append(300)
            elif node_data.get('hardware_type') in ['input', 'output']:
                node_colors.append('#AA00FF')  # 紫红色 - 输入输出
                node_sizes.append(350)
            elif node_data.get('type') == 'bitvector_constant':
                node_colors.append('#00AAFF')  # 浅蓝色 - 位向量常量
                node_sizes.append(250)
            elif node_data.get('type') == 'identifier':
                node_colors.append('#FFAAFF')  # 粉色 - 标识符
                node_sizes.append(250)
            # 在 visualize_tree 方法的颜色设置部分添加：
            elif node_data.get('type') == 'function_reference':
                node_colors.append('#FF6600')  # 橙红色 - 函数引用
                node_sizes.append(300)
            # 在可视化的节点颜色设置部分修改：
            elif node_data.get('is_assertion_root') or node_data.get('name', '').startswith('assertion_'):
                node_colors.append('#FF00FF')  # 品红色 - 断言节点
                node_sizes.append(400)
            else:
                node_colors.append('#AAAAAA')  # 灰色 - 其他
                node_sizes.append(200)
        
        # 绘制边
        nx.draw_networkx_edges(subgraph, pos,
                            edge_color='#666666',
                            arrows=True,
                            arrowsize=20,
                            arrowstyle='->',
                            alpha=0.6,
                            width=1.5)
        
        # 绘制节点
        nx.draw_networkx_nodes(subgraph, pos,
                            node_color=node_colors,
                            node_size=node_sizes,
                            alpha=0.8,
                            linewidths=2,
                            edgecolors='black')
        
        # 添加节点标签
        labels = {}
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            
            # 优化标签显示
            if node_data.get('type') == 'module':
                label = f"MODULE\n{node_data.get('name', node)}"
            elif node_data.get('is_function_root'):
                func_name = node_data.get('name', node)
                if len(func_name) > 15:
                    func_name = func_name[:12] + '...'
                label = f"FUNC\n{func_name}"
            elif node_data.get('type') == 'compound':
                operator = node_data.get('operator', 'op')
                arity = node_data.get('arity', 0)
                label = f"{operator}\n({arity})"
            elif node_data.get('hardware_type'):
                hw_type = node_data.get('hardware_type', '').upper()
                name = node_data.get('name', node_data.get('identifier', node))
                if len(name) > 10:
                    name = name[:8] + '..'
                width = node_data.get('width', node_data.get('bit_width', ''))
                if width:
                    label = f"{hw_type}\n{name}[{width}]"
                else:
                    label = f"{hw_type}\n{name}"
            elif node_data.get('type') == 'bitvector_constant':
                value = node_data.get('value', '')
                if len(value) > 8:
                    value = value[:6] + '..'
                label = f"BV\n{value}"
            elif node_data.get('type') == 'identifier':
                identifier = node_data.get('identifier', node)
                if len(identifier) > 10:
                    identifier = identifier[:8] + '..'
                label = f"ID\n{identifier}"
            # 在标签设置部分添加：
            elif node_data.get('type') == 'function_reference':
                ref_name = node_data.get('referenced_function', node)
                if len(ref_name) > 10:
                    ref_name = ref_name[:8] + '..'
                label = f"→{ref_name}"  # 使用箭头表示引用
            elif node_data.get('type') == 'function_call':
                call_name = node_data.get('referenced_function', node)
                if len(call_name) > 10:
                    call_name = call_name[:8] + '..'
                label = f"call\n{call_name}"

            elif node_data.get('is_assertion_root') or node_data.get('name', '').startswith('assertion_'):
                if node_data.get('is_assertion_root'):
                    assert_index = node_data.get('assertion_index', 'X')
                    label = f"ASSERT\n#{assert_index}"
                else:
                    assert_name = node_data.get('name', node)
                    label = f"ASSERT\n{assert_name.replace('assertion_', '#')}"

            else:
                name = node_data.get('name', node_data.get('symbol', node))
                if len(name) > 10:
                    name = name[:8] + '..'
                label = name
            
            labels[node] = label

        nx.draw_networkx_labels(subgraph, pos, labels, 
                            font_size=8, 
                            font_weight='bold',
                            font_family='monospace')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF4444', markersize=15, label='Module'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4444FF', markersize=12, label='Function'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6600', markersize=10, label='Func Ref'),  # 新增
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF00FF', markersize=10, label='Assertion'),  # 新增
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00AA00', markersize=10, label='Compound Op'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFAA00', markersize=10, label='Register'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFFF00', markersize=10, label='Wire'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#AA00FF', markersize=10, label='I/O Port'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00AAFF', markersize=8, label='Constant'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFAAFF', markersize=8, label='Identifier')
        ]
        
        plt.legend(handles=legend_elements, 
                loc='upper left', 
                bbox_to_anchor=(0.02, 0.98),
                fontsize=10)
        
        plt.title("Cache Control SMT Tree Structure (Hierarchical Layout)", 
                fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        
        # 调整布局以适应标签
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            print(f'图形已保存到 {save_path}')
        
        plt.show()



    def _parse_assertions(self, content):
        """解析assert语句 - 使用行扫描方法"""
        
        lines = content.split('\n')
        i = 0
        assertion_count = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('(assert'):
                # 找到assert的开始
                assertion_lines = [line]
                bracket_count = line.count('(') - line.count(')')
                i += 1
                
                # 继续读取直到括号平衡
                while i < len(lines) and bracket_count > 0:
                    line = lines[i].strip()
                    if line:  # 跳过空行
                        assertion_lines.append(line)
                        bracket_count += line.count('(') - line.count(')')
                    i += 1
                
                # 解析这个断言
                if self._parse_single_assertion_from_lines(assertion_lines, assertion_count):
                    assertion_count += 1
            else:
                i += 1
        
        # print(f"Total assertions found: {assertion_count}")
        return assertion_count

    def _parse_single_assertion_from_lines(self, lines, assertion_index):
        """从行列表中解析单个断言"""
        if not lines:
            return False
        
        # 合并所有行
        full_text = ' '.join(lines)
        
        try:
            # 提取断言体
            assertion_body = self._extract_assertion_body(full_text)
            
            if not assertion_body:
                return False
            
            # 解析断言表达式
            assert_node = self._parse_expression(assertion_body, f"assertion_{assertion_index}")
            
            # 为断言根节点添加特殊标识
            if assert_node and assert_node in self.tree.nodes:
                self.tree.nodes[assert_node]['is_assertion_root'] = True
                self.tree.nodes[assert_node]['assertion_index'] = assertion_index
            
            self.assertions[f"assertion_{assertion_index}"] = {
                'node_id': assert_node,
                'body': assertion_body,
                'full_definition': full_text
            }
            
            return True
            
        except Exception as e:
            print(f"Error parsing assertion {assertion_index}: {e}")
            return False


    def _extract_assertion_body(self, full_text):
        """提取断言体"""
        content = full_text.strip()
        if not content.startswith('(assert'):
            return None
        
        # 去掉最外层的括号和assert关键字
        content = content[7:].strip()  # 去掉 "(assert"
        if content.endswith(')'):
            content = content[:-1].strip()  # 去掉最后的 ")"
        
        return content


    def _parse_declarations(self, content):
        """解析declare-fun和declare-sort语句"""

        sort_pattern = r'\(declare-sort\s+([^)]+)\s+(\d+)\)'
        for match in re.finditer(sort_pattern, content):
            sort_name = match.group(1).strip('|')
            arity = int(match.group(2))
            self.declarations[sort_name] = {'type': 'sort', 'arity': arity}

        fun_pattern = r'\(declare-fun\s+([^)]+)\s+\(([^)]*)\)\s+([^)]+)\)'

        for match in re.finditer(fun_pattern, content):
            fun_name = match.group(1).strip("|")
            params = match.group(2).strip()
            return_type = match.group(3).strip()

            self.declarations[fun_name] = {
                'type': 'function',
                'params': params.split() if params else [],
                'return_type': return_type
            }

    def _parse_define_functions(self, content):
        """使用行扫描方法解析define-fun语句"""
        
        lines = content.split('\n')
        i = 0
        function_count = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            if line.startswith('(define-fun'):
                # 找到define-fun的开始
                function_lines = [line]
                bracket_count = line.count('(') - line.count(')')
                i += 1
                
                # 继续读取直到括号平衡
                while i < len(lines) and bracket_count > 0:
                    line = lines[i].strip()
                    if line:  # 跳过空行
                        function_lines.append(line)
                        bracket_count += line.count('(') - line.count(')')
                    i += 1
                
                # 解析这个函数定义
                if self._parse_single_function_from_lines(function_lines):
                    function_count += 1
            else:
                i += 1
        
        # print(f"Total functions found: {function_count}")
        return function_count

    def _parse_single_function_from_lines(self, lines):
        """从行列表中解析单个函数定义"""
        if not lines:
            return False
        
        # 合并所有行
        full_text = ' '.join(lines)
        
        try:
            # 解析函数定义的各个部分
            func_name, params, return_type, body = self._extract_function_parts(full_text)
            
            if not func_name:
                return False
            
            # 为函数创建节点ID
            func_node_id = f"node_{self.node_counter}"
            self.node_counter += 1
            
            # 存储函数信息
            self.functions[func_name] = {
                'node_id': func_node_id,  # 使用实际的节点ID
                'params': params,
                'return_type': return_type,
                'body': body,
                'original_name': func_name,
                'full_definition': full_text
            }
            
            # 将函数节点添加到树中
            self.tree.add_node(func_node_id,
                            type='function',
                            name=func_name,
                            params=params,
                            return_type=return_type,
                            is_function_root=True)
            
            # 如果函数体不为空，解析函数体并创建子树
            if body and body.strip():
                self._parse_expression(body, func_name, func_node_id, is_function_root=True)
            
            return True
            
        except Exception as e:
            print(f"Error parsing function: {e}")
            print(f"Function text: {full_text[:100]}...")
            return False


    def _extract_function_parts(self, full_text):
        """提取函数定义的各个部分"""
        # 格式: (define-fun |function_name| (params) return_type body)
        
        content = full_text.strip()
        if not content.startswith('(define-fun'):
            return None, None, None, None
        
        # 去掉最外层的括号和define-fun关键字
        content = content[11:].strip()  # 去掉 "(define-fun"
        if content.endswith(')'):
            content = content[:-1].strip()  # 去掉最后的 ")"
        
        # 提取函数名
        func_name, remaining = self._extract_function_name(content)
        if not func_name:
            return None, None, None, None
        
        # 提取参数列表
        params, remaining = self._extract_parameters(remaining)
        
        # 提取返回类型和函数体
        return_type, body = self._extract_return_type_and_body(remaining)
        
        return func_name, params, return_type, body

    def _extract_function_name(self, content):
        """提取函数名"""
        content = content.strip()
        
        if content.startswith('|'):
            # 处理带|符号的函数名
            end_pos = content.find('|', 1)
            if end_pos == -1:
                return None, content
            func_name = content[1:end_pos]  # 去掉|符号
            remaining = content[end_pos + 1:].strip()
            return func_name, remaining
        else:
            # 处理普通函数名
            # 找到第一个空格或括号
            for i, char in enumerate(content):
                if char in ' \t\n(':
                    func_name = content[:i]
                    remaining = content[i:].strip()
                    return func_name, remaining
            
            # 如果没找到分隔符，整个就是函数名
            return content, ""

    def _extract_parameters(self, content):
        """提取参数列表"""
        content = content.strip()
        
        if not content.startswith('('):
            return "", content
        
        # 找到匹配的右括号
        bracket_count = 0
        i = 0
        
        for i, char in enumerate(content):
            if char == '(':
                bracket_count += 1
            elif char == ')':
                bracket_count -= 1
                if bracket_count == 0:
                    break
        
        if bracket_count == 0:
            params = content[1:i]  # 去掉括号
            remaining = content[i+1:].strip()
            return params.strip(), remaining
        else:
            # 括号不匹配，返回空参数
            return "", content

    def _extract_return_type_and_body(self, content):
        """提取返回类型和函数体"""
        content = content.strip()
        
        if not content:
            return "", ""
        
        # 尝试找到返回类型（通常是第一个token）
        tokens = self._tokenize_smt(content)
        
        if len(tokens) == 0:
            return "", ""
        elif len(tokens) == 1:
            # 只有一个token，可能是返回类型或者函数体
            if tokens[0].startswith('('):
                return "", tokens[0]
            else:
                return tokens[0], ""
        else:
            # 第一个token是返回类型，其余是函数体
            return_type = tokens[0]
            body_tokens = tokens[1:]
            body = ' '.join(body_tokens)
            return return_type, body

    def _tokenize_smt(self, content):
        """简单的SMT表达式分词器"""
        tokens = []
        current_token = ""
        bracket_count = 0
        in_token = False
        
        i = 0
        while i < len(content):
            char = content[i]
            
            if char in ' \t\n' and bracket_count == 0 and not in_token:
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                i += 1
                continue
            
            if char == '(':
                if bracket_count == 0 and current_token:
                    # 当前token结束，开始新的括号表达式
                    tokens.append(current_token)
                    current_token = ""
                current_token += char
                bracket_count += 1
                in_token = True
            elif char == ')':
                current_token += char
                bracket_count -= 1
                if bracket_count == 0:
                    tokens.append(current_token)
                    current_token = ""
                    in_token = False
            else:
                current_token += char
                if bracket_count == 0:
                    in_token = True
            
            i += 1
        
        if current_token:
            tokens.append(current_token)
        
        return tokens

    def _parse_yosys_annotations(self, content):
        """解析Yosys特有的注释信息"""
        
        # 解析yosys-smt2-register
        reg_pattern = r'; yosys-smt2-register\s+([^\s]+)\s+(\d+)'
        for match in re.finditer(reg_pattern, content):
            reg_name = match.group(1)
            width = int(match.group(2))
            self.registers[reg_name] = {'width': width, 'type': 'register'}
        
        # 解析yosys-smt2-wire
        wire_pattern = r'; yosys-smt2-wire\s+([^\s]+)\s+(\d+)'
        for match in re.finditer(wire_pattern, content):
            wire_name = match.group(1)
            width = int(match.group(2))
            self.wires[wire_name] = {'width': width, 'type': 'wire'}
        
        # 解析yosys-smt2-input
        input_pattern = r'; yosys-smt2-input\s+([^\s]+)\s+(\d+)'
        for match in re.finditer(input_pattern, content):
            input_name = match.group(1)
            width = int(match.group(2))
            self.inputs[input_name] = {'width': width, 'type': 'input'}
        
        # 解析yosys-smt2-output
        output_pattern = r'; yosys-smt2-output\s+([^\s]+)\s+(\d+)'
        for match in re.finditer(output_pattern, content):
            output_name = match.group(1)
            width = int(match.group(2))
            self.outputs[output_name] = {'width': width, 'type': 'output'}
   
   

    def _parse_expression(self, expr, name=None, parent_id=None, is_function_root=False):
        """使用迭代方式解析表达式并构建树，避免递归深度限制"""
        
        # 使用栈来模拟递归调用
        # 栈中的每个元素是一个字典，包含处理所需的所有信息
        stack = [{
            'expr': expr.strip(),
            'name': name,
            'parent_id': parent_id,
            'is_function_root': is_function_root,
            'node_id': None,  # 将在处理时分配
            'state': 'create_node'  # 处理状态：'create_node', 'process_children', 'complete'
        }]
        
        # 存储已创建的节点，用于返回根节点ID
        created_nodes = {}
        root_node_id = None
        
        while stack:
            current = stack.pop()
            expr = current['expr']
            name = current['name']
            parent_id = current['parent_id']
            is_function_root = current['is_function_root']
            state = current['state']
            
            if state == 'create_node':
                # 检查是否是函数引用（避免重复展开）
                if expr.startswith('|') and expr.endswith('|'):
                    func_ref_name = expr[1:-1]
                    if func_ref_name in self.functions and not is_function_root:
                        # 这是一个函数引用，创建引用节点而不是展开
                        node_id = f"node_{self.node_counter}"
                        self.node_counter += 1
                        
                        node_info = {
                            'type': 'function_reference',
                            'name': func_ref_name,
                            'referenced_function': func_ref_name,
                            'depth': 0 if parent_id is None else self.tree.nodes[parent_id].get("depth", 0) + 1,
                            'display_name': f"→{func_ref_name}"
                        }
                        
                        self.tree.add_node(node_id, **node_info)
                        
                        if parent_id is not None:
                            self.tree.add_edge(parent_id, node_id, edge_type='function_call')
                        
                        created_nodes[id(current)] = node_id
                        if root_node_id is None:
                            root_node_id = node_id
                        continue
                
                # 检查是否是已经处理过的函数调用
                if expr in self.functions and not is_function_root:
                    # 直接创建函数引用，不展开
                    node_id = f"node_{self.node_counter}"
                    self.node_counter += 1
                    
                    node_info = {
                        'type': 'function_call',
                        'name': expr,
                        'referenced_function': expr,
                        'depth': 0 if parent_id is None else self.tree.nodes[parent_id].get("depth", 0) + 1,
                        'display_name': f"call:{expr}"
                    }
                    
                    self.tree.add_node(node_id, **node_info)
                    
                    if parent_id is not None:
                        self.tree.add_edge(parent_id, node_id, edge_type='function_call')
                    
                    created_nodes[id(current)] = node_id
                    if root_node_id is None:
                        root_node_id = node_id
                    continue

                # 创建新节点
                node_id = f"node_{self.node_counter}"
                self.node_counter += 1
                current['node_id'] = node_id
                
                # 基本节点信息
                node_info = {
                    'expression': expr[:100] + '...' if len(expr) > 100 else expr,
                    'name': name or f"expr_{self.node_counter}",
                    "is_function_root": is_function_root,
                    "depth": 0 if parent_id is None else self.tree.nodes[parent_id].get("depth", 0) + 1
                }

                if expr.startswith('('):
                    # 复合表达式
                    node_info['type'] = 'compound'

                    # 提取操作符和参数
                    inner = expr[1:-1].strip()
                    parts = self._split_s_expression(inner)

                    if parts:
                        operator = parts[0]
                        args = parts[1:]
                        
                        node_info['operator'] = operator
                        node_info['arity'] = len(args)

                        # 特殊处理不同的操作符
                        if operator == 'ite':
                            node_info['operator_type'] = 'conditional'
                        elif operator in ['=', '>', '<', '>=', '<=']:
                            node_info['operator_type'] = 'comparison'
                        elif operator in ['and', 'or', 'not']:
                            node_info['operator_type'] = 'logical'
                        elif operator in ['bvand', 'bvor', 'bvnot', 'bvxor']:
                            node_info['operator_type'] = 'bitvector_logical'
                        elif operator in ['bvadd', 'bvsub', 'bvmul']:
                            node_info['operator_type'] = 'bitvector_arithmetic'
                        elif operator == 'concat':
                            node_info['operator_type'] = 'concatenation'
                        elif operator == '_ extract' or (len(parts) >= 2 and parts[0] == '_' and parts[1] == 'extract'):
                            node_info['operator_type'] = 'extraction'
                            # 处理 (_ extract high low) 格式
                            if len(parts) >= 4:
                                node_info['high_bit'] = parts[2]
                                node_info['low_bit'] = parts[3]
                                # 重新组织参数，跳过 _ extract high low
                                args = parts[4:]
                                node_info['arity'] = len(args)
                        elif operator == '_ zero_extend':
                            node_info['operator_type'] = 'zero_extend'
                            if len(parts) >= 3:
                                node_info['extend_bits'] = parts[2]
                                args = parts[3:]
                                node_info['arity'] = len(args)
                        else:
                            node_info['operator_type'] = 'function_application'

                        self.tree.add_node(node_id, **node_info)

                        if parent_id is not None:
                            self.tree.add_edge(parent_id, node_id, edge_type='child')

                        # 准备处理子节点
                        current['state'] = 'process_children'
                        current['args'] = args
                        current['processed_args'] = 0
                        stack.append(current)  # 重新放入栈中等待处理子节点
                        
                        # 将子节点处理任务加入栈（逆序加入，保证正确的处理顺序）
                        for i in reversed(range(len(args))):
                            arg = args[i]
                            child_task = {
                                'expr': arg,
                                'name': f"{name}_arg{i}" if name else None,
                                'parent_id': node_id,
                                'is_function_root': False,
                                'node_id': None,
                                'state': 'create_node'
                            }
                            stack.append(child_task)
                    else:
                        # 没有参数的复合表达式
                        self.tree.add_node(node_id, **node_info)
                        if parent_id is not None:
                            self.tree.add_edge(parent_id, node_id, edge_type='child')

                else:
                    # 原子表达式处理
                    if expr.startswith('#b'):
                        node_info['type'] = 'bitvector_constant'
                        node_info['value'] = expr[2:]
                        node_info['bit_width'] = len(expr[2:])
                    elif expr.startswith('#x'):
                        node_info['type'] = 'hex_constant'
                        node_info['value'] = expr[2:]
                    elif expr.startswith('|') and expr.endswith('|'):
                        node_info['type'] = 'identifier'
                        node_info['identifier'] = expr[1:-1]
                        # 检查是否是已知的寄存器、线网等
                        clean_name = expr[1:-1]
                        if clean_name in self.registers:
                            node_info['hardware_type'] = 'register'
                            node_info.update(self.registers[clean_name])
                        elif clean_name in self.wires:
                            node_info['hardware_type'] = 'wire'
                            node_info.update(self.wires[clean_name])
                        elif clean_name in self.inputs:
                            node_info['hardware_type'] = 'input'
                            node_info.update(self.inputs[clean_name])
                        elif clean_name in self.outputs:
                            node_info['hardware_type'] = 'output'
                            node_info.update(self.outputs[clean_name])
                    elif expr.isdigit() or (expr.startswith('-') and expr[1:].isdigit()):
                        node_info['type'] = 'integer_constant'
                        node_info['value'] = int(expr)
                    elif expr in ['true', 'false']:
                        node_info['type'] = 'boolean_constant'
                        node_info['value'] = expr == 'true'
                    else:
                        node_info['type'] = 'symbol'
                        node_info['symbol'] = expr
                    
                    self.tree.add_node(node_id, **node_info)
                    
                    # 如果有父节点，添加边
                    if parent_id is not None:
                        self.tree.add_edge(parent_id, node_id, edge_type='child')
                
                created_nodes[id(current)] = node_id
                if root_node_id is None:
                    root_node_id = node_id
                    
            elif state == 'process_children':
                # 子节点已经处理完毕，当前节点处理完成
                node_id = current['node_id']
                created_nodes[id(current)] = node_id
                if root_node_id is None:
                    root_node_id = node_id
        
        return root_node_id




    def _split_s_expression(self, expr):
        """分割S表达式为操作符和参数"""
        parts = []
        current = ""
        paren_count = 0
        in_quotes = False
        
        i = 0
        while i < len(expr):
            char = expr[i]
            
            if char == '|' and not in_quotes:
                in_quotes = True
                current += char
            elif char == '|' and in_quotes:
                in_quotes = False
                current += char
            elif char == '(' and not in_quotes:
                paren_count += 1
                current += char
            elif char == ')' and not in_quotes:
                paren_count -= 1
                current += char
            elif char == ' ' and paren_count == 0 and not in_quotes:
                if current.strip():
                    parts.append(current.strip())
                    current = ""
            else:
                current += char
            
            i += 1
        
        if current.strip():
            parts.append(current.strip())
        
        return parts


    def _build_main_tree(self):
        """构建主要的模块树结构"""
        module_root = f"node_{self.node_counter}"
        self.node_counter += 1
        self.tree.add_node(module_root, 
                          type='module',
                          name='Cache_Ctrl',
                          depth=0)
        
        for func_name, func_info in self.functions.items():
            self.tree.add_edge(module_root, func_info['node_id'], 
                             edge_type='contains')
            
        # 连接断言到模块根节点
        for assert_name, assert_info in self.assertions.items():
            self.tree.add_edge(module_root, assert_info['node_id'], 
                             edge_type='asserts')


 
    def _get_statistics(self):
        """获取树的统计信息"""
        
        # 获取所有节点的深度值
        depths = [data.get("depth", 0) for _, data in self.tree.nodes(data=True)]
        max_depth = max(depths) if depths else 0


        # 统计不同类型的节点
        node_type_counts = defaultdict(int)
        for _, data in self.tree.nodes(data=True):
            node_type = data.get('type', 'unknown')
            node_type_counts[node_type] += 1
            
        stats = {
            "total_nodes": len(self.tree.nodes()),
            "total_edges": len(self.tree.edges()),
            "num_functions": len(self.functions),
            "num_assertions": len(self.assertions),  # 添加这一行
            "num_registers": len(self.registers),
            "num_wires": len(self.wires),
            "num_inputs": len(self.inputs),
            'num_outputs': len(self.outputs),
            "max_depth": max_depth,
            "node_type_counts": dict(node_type_counts)  # 添加节点类型统计
        }

        operator_counts = defaultdict(int)

        for _, data in self.tree.nodes(data=True):
            if "operator" in data:
                operator_counts[data["operator"]] += 1

        stats["operator_counts"] = dict(operator_counts)
        
        return stats

            


    def _create_hierarchical_layout(self, graph):
        """创建自定义的分层布局"""
        
        # 找到根节点（入度为0的节点）
        root_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        
        if not root_nodes:
            # 如果没有根节点，选择度数最大的节点作为根
            root_nodes = [max(graph.nodes(), key=lambda x: graph.degree(x))]
        
        # 计算每个节点的层级
        levels = {}
        visited = set()
        
        def assign_level(node, level):
            if node in visited:
                return
            visited.add(node)
            levels[node] = max(levels.get(node, 0), level)
            
            for successor in graph.successors(node):
                assign_level(successor, level + 1)
        
        # 从每个根节点开始分配层级
        for root in root_nodes:
            assign_level(root, 0)
        
        # 为没有被访问的节点分配层级
        for node in graph.nodes():
            if node not in levels:
                levels[node] = 0
        
        # 按层级组织节点
        level_nodes = {}
        for node, level in levels.items():
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)
        
        # 计算位置
        pos = {}
        max_level = max(level_nodes.keys()) if level_nodes else 0
        
        for level, nodes in level_nodes.items():
            y = max_level - level  # 从上到下
            num_nodes = len(nodes)
            
            if num_nodes == 1:
                x_positions = [0]
            else:
                # 在水平方向上均匀分布节点
                x_positions = np.linspace(-num_nodes/2, num_nodes/2, num_nodes)
            
            for i, node in enumerate(nodes):
                pos[node] = (x_positions[i], y)
        
        return pos



    # 使用示例
def main():
    # 解析SMT文件
    parser = YosysSMTtoTree()
    tree = parser.parse_smt_file('TestCase.smt2')
    
    # 打印摘要
    parser.print_summary()
    
    # 提取特征
    features = parser.extract_features()
    print(f"\n=== 特征提取结果 ===")
    print(f"节点特征矩阵形状: {features['node_features'].shape}")
    print(f"边索引矩阵形状: {features['edge_index'].shape}")
    print(f"边属性矩阵形状: {features['edge_attr'].shape}")
    
    # 统计信息
    stats = features['statistics']
    print(f"\n=== 统计信息 ===")
    for key, value in stats.items():
        if key != 'operator_distribution':
            print(f"{key}: {value}")
    
    print(f"\n=== 操作符分布 ===")
    for op, count in sorted(stats['operator_counts'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{op}: {count}")
    
    # 可视化（可选）
    parser.visualize_tree(max_nodes=50, save_path='TestCase.png')
    
    return parser, features
        
        

        

if __name__ == "__main__":
    parser, features = main()
    
'''
每个节点的特征 features 是一个长度为 20 的列表（即20维），具体如下：

是否是复合节点（compound）
是否是位向量常量（bitvector_constant）
是否是标识符（identifier）
是否是符号（symbol）
是否是模块（module）
是否是函数根节点（is_function_root）
是否是函数引用（function_reference）
是否是函数调用（function_call）
是否是断言节点（is_assertion_root 或 name 以 assertion_ 开头）
是否是寄存器（hardware_type == 'register'）
是否是线网（hardware_type == 'wire'）
是否是输入端口（hardware_type == 'input'）
是否是输出端口（hardware_type == 'output'）
节点的arity（参数个数）
节点的bit_width（比特宽度）
节点的width（宽度）
节点的depth（深度）
节点的度数（degree）
节点的前驱个数（predecessors）
节点的后继个数（successors）



每条边的属性是一个长度为 5 的列表，具体如下：

是否是child边
是否是contains边
是否是asserts边
是否是function_call边
默认权重（始终为1）
'''