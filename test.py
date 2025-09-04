import torch

if torch.cuda.device_count() > 1:
        print(f"使用{torch.cuda.device_count()}个GPU进行训练")
        # model = torch.nn.DataParallel(model)