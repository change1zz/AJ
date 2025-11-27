import argparse
import csv
import os
import time

import numpy as np


def read_adj_csv(path):
    rows = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append([int(float(x)) for x in row])
    if not rows:
        raise ValueError(f'空的邻接矩阵: {path}')
    return np.array(rows, dtype=int)


def analyze_components(adj):
    n = adj.shape[0]
    visited = set()
    comps = []
    for i in range(n):
        if i in visited:
            continue
        stack = [i]
        visited.add(i)
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            neighbors = np.where(adj[v] > 0)[0]
            for nb in neighbors:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def evaluate_sample(sample_dir):
    adj_path = os.path.join(sample_dir, 'comm_adj.csv')
    if not os.path.isfile(adj_path):
        raise FileNotFoundError(f'缺少 comm_adj.csv: {sample_dir}')
    adj = read_adj_csv(adj_path)
    binary_adj = (adj > 0).astype(int)
    degrees = binary_adj.sum(axis=1)
    comps = analyze_components(binary_adj)
    run_args_path = os.path.join(sample_dir, 'run_args.json')
    metrics = {
        'sample': os.path.basename(sample_dir),
        'nodes': adj.shape[0],
        'avg_degree': float(np.mean(degrees)),
        'min_degree': int(np.min(degrees)),
        'max_degree': int(np.max(degrees)),
        'component_count': len(comps),
        'largest_component': max(len(c) for c in comps),
        'is_connected': int(len(comps) == 1),
    }
    if os.path.isfile(run_args_path):
        metrics['has_run_args'] = 1
    else:
        metrics['has_run_args'] = 0
    return metrics


def main():
    parser = argparse.ArgumentParser(description='统计拓扑数据集中每个 sample 的连通性与度信息')
    parser.add_argument('--dataset', type=str,
                        default=os.path.join('..', 'datasets', 'topologies', '16_nodes', '3d_2025-10-28-17-26-47'),
                        help='拓扑数据集根目录（包含 sample_xxx 子目录）')
    parser.add_argument('--limit', type=int, default=-1, help='最多评估的 sample 数量；-1 表示全部')
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f'数据集路径不存在：{dataset_dir}')
    samples = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir)
               if d.startswith('sample_') and os.path.isdir(os.path.join(dataset_dir, d))]
    samples.sort()
    if args.limit > 0:
        samples = samples[:args.limit]

    os.makedirs('result', exist_ok=True)
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    out_csv = os.path.join('result', f'topology_eval_{timestamp}.csv')

    records = []
    for sample in samples:
        try:
            metrics = evaluate_sample(sample)
            records.append(metrics)
            print(f"[Eval] {metrics['sample']}: connected={metrics['is_connected']} avg_deg={metrics['avg_degree']:.2f} comps={metrics['component_count']}")
        except Exception as e:
            print(f"[Eval] 跳过 {sample}: {e}")

    if not records:
        print("[Eval] 未生成任何统计数据。")
        return

    headers = ['sample', 'nodes', 'avg_degree', 'min_degree', 'max_degree',
               'component_count', 'largest_component', 'is_connected', 'has_run_args']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    print(f'[Eval] 统计结果已写入：{out_csv}')


if __name__ == '__main__':
    main()
