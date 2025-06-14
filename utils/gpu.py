import os

import torch.cuda


def get_device(gpu_num):
    if gpu_num is None:
        return GPU.auto_choose(torch_format=True)

    if gpu_num == -1:
        print('Choosing CPU device')
        return 'cpu'

    print(f'Choosing {gpu_num}-th GPU')
    return f'CUDA: {gpu_num}'


class GPU:
    @classmethod
    def parse_gpu_info(cls, line, args):
        def to_number(v):
            return float(v.upper().strip().replace('MIB', '').replace('W', ''))

        def processor(k, v):
            return (int(to_number(v)) if 'Not Support' not in v else 1) if k in params else v.strip()

        params = ['memory.free', 'memory.total', 'power.draw', 'power.limit']
        return {k: processor(k, v) for k, v in zip(args, line.strip().split(','))}

    @classmethod
    def get_gpus(cls):
        args = ['index', 'gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(args))
        results = os.popen(cmd).readlines()
        return [cls.parse_gpu_info(line, args) for line in results]

    @classmethod
    def auto_choose(cls, torch_format=False):
        if not torch.cuda.is_available():
            print('system does not support CUDA')
            if torch_format:
                print('auto switching to CPU device')
                return "cpu"
            return -1

        gpus = cls.get_gpus()
        chosen_gpu = sorted(gpus, key=lambda d: d['memory.free'], reverse=True)[0]
        print(f'choosing {chosen_gpu["index"]}-th GPU with {chosen_gpu["memory.free"]} / {chosen_gpu["memory.total"]} MB')
        if torch_format:
            return "cuda:" + str(chosen_gpu['index'])
        return int(chosen_gpu['index'])
