

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
    all_num = 0
    for k, v in model.named_parameters():
        shape = str([_ for _ in v.shape])
        all_num+=v.numel()
        module = type(model.get_submodule(".".join(k.split(".")[:-1])))
        type_name = module.__module__+"."+module.__name__
        print(f"{k:-<66s}--{type_name:-<50}---{str(v.numel()):->15s}---{shape:->20s}---{str(v.dtype):<15s}")
    if hasattr(model, "get_memory_footprint"):
        mem = model.get_memory_footprint()
        print(f"Total num: {all_num} is {all_num/10**9} Billion. GPU mem: {mem/1024**3} GB")
    else:
        print(f"Total num: {all_num} is {all_num/10**9} Billion")
    print("End.", words,"\n\n")
    
