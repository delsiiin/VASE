import argparse
import copy

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--outdir', type=str, default='0')
parser.add_argument('--type', type=str, default='vicuna')
parser.add_argument('--basemodel', type=str, default='/root/MODELS/vicuna-7b-v1.3')
parser.add_argument('--data', type=str, default='/root/idea/speculative_decoding/sharegpt_medusa/ShareGPT_V4.3_unfiltered_cleaned_split.json')
args = parser.parse_args()

import os
from concurrent.futures import ThreadPoolExecutor

s = 0
e = 68000 - 1
#e = 68 - 1
#gpus = [[0],[1],[2],[3],[4],[5],[6],[7]]

gpus=[[0],[1]]
num_p = len(gpus)
outdir = '{}/sharegpt_{}_{}_mufp16'.format(args.outdir,s,e)


def split_range(start, end, n, over=False):
    length = end - start + 1  # Include the end
    base_interval = length // n
    additional = length % n  # Get the remainder of the division
    intervals = []
    previous = start

    for i in range(n):
        current_interval = base_interval + (1 if i < additional else 0)
        if over:
            intervals.append((previous, previous + current_interval))
        else:
            intervals.append((previous, previous + current_interval - 1))  # '-1' because the end is inclusive
        previous += current_interval

    return intervals


def run_command(cmd):
    os.system(cmd)


if not os.path.exists(outdir):
    os.makedirs(outdir)


data_a = split_range(s, e, num_p, over=True)
commands = []
for i in range(num_p):
    index = i
    start = data_a[i][0]
    end = data_a[i][1]
    # gpu_index_str = [str(i) for i in gpu_index]
    # gpu_index_str=','.join(gpu_index_str)
    gpu_index = gpus[i]
    gpu_index_str = ' '.join(map(str, gpu_index))
    # gpu_index_str='['+gpu_index_str+']'
    if args.type == 'vicuna':
        command = "python gen_data_all_vicuna.py --start={} --end={} --index={} --gpu_index {} --outdir {} --basemodel {} --data {}".format(start, end, index,gpu_index_str,outdir,args.basemodel,args.data)
    elif args.type == 'llama2':
        command = "python gen_data_all_llama2.py --start={} --end={} --index={} --gpu_index {} --outdir {} --basemodel {} --data {}".format(start, end, index,gpu_index_str,outdir,args.basemodel,args.data)
    elif args.type == 'llama3':
        command = "python gen_data_all_llama3.py --start={} --end={} --index={} --gpu_index {} --outdir {} --basemodel {} --data {}".format(start, end, index,gpu_index_str,outdir,args.basemodel,args.data)
    commands.append(command)
# run_command(commands[0])
# commands=commands[:1]
with ThreadPoolExecutor(max_workers=len(commands)) as executor:
    for command in commands:
        executor.submit(run_command, command)
        print(command)