

def check_gpu_usage_torch():
    import torch
    # 查看当前可用的GPU设备数量
    device_count = torch.cuda.device_count()
    print("可用的GPU设备数量：", device_count)

    if device_count > 0:
        # 查看当前正在使用的GPU设备索引
        current_device = torch.cuda.current_device()
        print("当前正在使用的GPU设备索引：", current_device)
        for i in range(device_count):
            # 查看当前正在使用的GPU设备的名称
            device_name = torch.cuda.get_device_name(current_device)
            # 查看当前正在使用的GPU设备的内存占用情况
            memory_allocated = torch.cuda.memory_allocated(current_device) / 1024 ** 3
            # 查看当前正在使用的GPU设备的内存缓存情况
            memory_cached = torch.cuda.memory_cached(current_device) / 1024 ** 3
            print(f"CUDA:{i} --- {device_name} --- Allocated: {memory_allocated:.2f}GB --- Cached: {memory_cached:.2f}GB")
    else:
        print("没有可用的GPU设备。")


def check_gpu_usage_system():
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                 '--format=csv,nounits,noheader'], capture_output=True, text=True)
        output = result.stdout.strip().split('\n')
        if len(output) > 0:
            for line in output:
                memory_used, memory_total = line.split(',')
                memory_used = int(memory_used) / 1024
                memory_total = int(memory_total) / 1024
                memory_utilization = memory_used / memory_total * 100
                print(f"系统显存占用：{memory_used:.2f} GB / {memory_total:.2f} GB ({memory_utilization:.2f}%)")
        else:
            print("系统无法获取GPU占用情况。")
    except FileNotFoundError:
        print("未找到nvidia-smi命令，请确保NVIDIA驱动已正确安装。")


def check_gpu():
    check_gpu_usage_system()
    check_gpu_usage_torch()

def model_info(model, words=""):
    import torch
    assert isinstance(model, torch.nn.Module)
    print("##", words)
    print("="*100)

    param_num = 0
    param_mem = 0
    for param_name, param in model.named_parameters():
        module = type(model.get_submodule(".".join(param_name.split(".")[:-1])))
        type_name = module.__module__+"."+module.__name__
        if len(type_name)>60:
            type_name = f"{type_name[:30]}...{type_name[-30:]}"
            
        param_num+=param.numel()
        param_mem += param.nelement()*param.element_size()
        sparse_num = (param==0).sum().item()
        sparse_ratio = sparse_num/param.numel()
        shape_str = f"{str(param.dtype):-<12s}---sparsity={sparse_ratio:.3f}---[{', '.join([str(_) for _ in param.shape])}]={param.numel()}"
        print(f"Param : {type_name:-<66s}---{param_name:-<66s}---{shape_str}")
    # https://discuss.pytorch.org/t/gpu-memory-that-model-uses/56822/2
    buf_num = 0
    buf_mem = 0
    for buf_name, buf in model.named_buffers():
        module = type(model.get_submodule(".".join(buf_name.split(".")[:-1])))
        type_name = module.__module__+"."+module.__name__
        if len(type_name)>60:
            type_name = f"{type_name[:30]}...{type_name[-30:]}"
        
        buf_num+=buf.numel()
        buf_mem += buf.nelement()*buf.element_size()
        sparse_num = (buf==0).sum().item()
        sparse_ratio = sparse_num/buf.numel()
        shape_str = f"{str(buf.dtype):-<12s}---sparsity={sparse_ratio:.3f}---[{', '.join([str(_) for _ in buf.shape])}]={buf.numel()}"
        print(f"Buffer: {type_name:-<66s}---{buf_name:-<66s}---{shape_str}")
    
    print(f"Param  numel: {param_num} is {param_num/10**9} Billion. GPU mem: {param_mem/1024**3} GB")
    if buf_num>0:
        print(f"Buffer numel: {buf_num} is {buf_num/10**9} Billion. GPU mem: {buf_mem/1024**3} GB")
        total_num = param_num+buf_num
        total_mem = param_mem+buf_mem
        print(f"Total  numel: {total_num} is {total_num/10**9} Billion. GPU mem: {total_mem/1024**3} GB")

    print("End.", words,"\n\n")
    
