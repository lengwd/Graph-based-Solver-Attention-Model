class DynamicConfig:
    """
    A class to dynamically create instances of a given class with specified arguments.

    Usage:

    custom_cfg = DynamicConfig(CustomClass, arg1=value1, arg2=value2, ...)
    instance = custom_cfg.build()

    # When you want to add or remove arguments:
    custom_cfg.arg3 = value3   # Add an argument
    custom_cfg.remove('arg2')  # Remove an argument
    print(custom_cfg)          # You can print the current configuration
    instance_new = custom_cfg.build()  # Build a new instance with the updated arguments
    """
    def __init__(self, cls: type, **kwargs): # 函数定义 打包 函数使用 解包
        self.cls = cls
        for k, v in kwargs.items():
            setattr(self, k, v) 
            # setattr 是 Python 的内置函数，功能是 「给对象动态添加一个属性」，语法是 setattr(对象, 属性名, 属性值)。

    def build(self):
        """
        Build an instance of the class with the current arguments.
        :return: An instance of the class with the current arguments.
        """
        kwargs = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and k != "cls"
        }
        return self.cls(**kwargs)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def remove(self, key: str) -> None:
        """
        Remove an attribute from the instance.
        :param key: The name of the attribute to remove.
        :return: None
        """
        if hasattr(self, key):
            delattr(self, key)


    def __str__(self):
        args = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and k != "cls"
        }
        return f"Config of {self.cls.__name__}: {args}"


    def __repr__(self):
        return self.__str__()
    

    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
if __name__ == "__main__":
    class Student:
    
        def __init__(self, name, age, major):
            self.name = name
            self.age = age
            self.major = major
            
        def introduce(self):
            return f"我叫{self.name}，{self.age}岁，专业是{self.major}"
        
    student_config = DynamicConfig(Student, name="张三", age=20)

    # 1. 初始化配置：指定目标类为Student，初始参数为name和age
    student_config = DynamicConfig(Student, name="张三", age=20)
    
    # 2. 查看初始配置
    print("初始配置:", student_config)
    # 输出: Config of Student: {'name': '张三', 'age': 20}
    
    # 3. 尝试创建实例（此时缺少major参数，会报错）
    try:
        student1 = student_config.build()
    except TypeError as e:
        print("创建实例失败:", e)  # 提示缺少major参数
    
    # 4. 使用add方法补充参数
    student_config.add(major="计算机科学")
    print("添加专业后:", student_config)
    # 输出: Config of Student: {'name': '张三', 'age': 20, 'major': '计算机科学'}
    
    # 5. 成功创建实例并调用方法
    student1 = student_config.build()
    print("学生介绍:", student1.introduce())
    # 输出: 我叫张三，20岁，专业是计算机科学
    
    # 6. 直接修改参数（改年龄）
    student_config.age = 21
    # 删除参数（暂时去掉专业）
    student_config.remove("major")
    print("修改并删除参数后:", student_config)
    # 输出: Config of Student: {'name': '张三', 'age': 21}
    
    # 7. 再次添加新参数（改专业为软件工程）
    student_config.add(major="软件工程")
    student2 = student_config.build()
    print("新学生介绍:", student2.introduce())
    # 输出: 我叫张三，21岁，专业是软件工程