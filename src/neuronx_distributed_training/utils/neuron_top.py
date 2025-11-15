import argparse
import inspect
import subprocess
import os
import glob
import json
import os
import re
import signal
import sys
import time

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


def launch(log_dir: Path) -> Path:
    nodename = os.uname().nodename
    node_log_dir = log_dir / "neuron_top_logs" / nodename

    script_path = inspect.getfile(inspect.currentframe())
    script_dir = Path(script_path).parent
    neuron_monitor_conf = script_dir / "neuron_monitor.conf"
    # run neuron-monitor -c neuron_monitor_conf | python neuron_top.py node_log_dir
    cmd = f"neuron-monitor -c {neuron_monitor_conf} | {sys.executable} {script_path} {node_log_dir}"
    subprocess.Popen(cmd, shell=True, start_new_session=True)
    return node_log_dir

class Const:
    MEM_USAGE_FIELD_NAMES_MODEL = [['tensors', 'constants', 'dma_buffers', 'application_memory'],
                                   ['tensors', 'constants', 'model_code', 'runtime_memory']]
    MEM_USAGE_FIELD_NAMES = [MEM_USAGE_FIELD_NAMES_MODEL[0],
                             MEM_USAGE_FIELD_NAMES_MODEL[1] + ['model_shared_scratchpad']]
    METRIC_GROUP_NAMES = ('neuroncore_counters', 'neuron_runtime_vcpu_usage', 'memory_used')
    NEURON_NODES_REGEX = re.compile(r'.*neuron([0-9]+)$')


class Utils:
    """ General use static functions
    """
    @staticmethod
    def human_readable_size(size, unit, div, right_align=True, fixed_width=True, precision=1):
        suffix = ('', 'K', 'M', 'G', 'T', 'P', 'E')
        remaining = float(size)
        format_spec = f'%.{precision}f%s'
        digit_count = len(str(div - 1))
        for suf in suffix:
            if remaining < div:
                if fixed_width:
                    if len(suf) == 0:
                        digit_count += 1
                    if right_align:
                        format_spec = f'%{digit_count + precision + 1}.{precision}f%s'
                    else:
                        format_spec = f'%-{digit_count + precision + 1}.{precision}f%s'
                return format_spec % (remaining, suf + unit)
            remaining = remaining / div
        return 'NaN'

    @staticmethod
    def human_readable_byte_size(size, right_align=True, fixed_width=True):
        return Utils.human_readable_size(size, 'B', 1024, right_align, fixed_width)

    @staticmethod
    def format_percentage(perc):
        return '  100%' if perc > 99.995 else '%5.2f%%' % (perc,)

    @staticmethod
    def format_flops(flops):
        tflops = flops * 1.0e-12
        if tflops < 0.001:
            return '       0TF'
        return '%8.3fTF' % (tflops)

    @staticmethod
    def prop_getter(arr, idx):
        return arr[idx] if idx < len(arr) else arr[0]

    @staticmethod
    def get_neuron_device_indices():
        indices = []
        file_matches = glob.glob('/dev/neuron*')
        for neuron_node in file_matches:
            result = re.search(Const.NEURON_NODES_REGEX, neuron_node)
            if result is None:
                continue
            indices.append(int(result.group(1)))
        return sorted(indices)

    @staticmethod
    def shorten_text(text, max_length, cut_left=True):
        if max_length <= 0:
            return ''
        overflow = len(text) - max_length
        if overflow <= 0:
            return text
        return text[overflow:] if cut_left else text[:-overflow]


def _init_aggregated_neuroncore_counters(destination):
    destination['error'] = ''
    destination['neuroncores_in_use'] = {}


def _aggregate_neuroncore_counters(destination, source):
    if source['error'] != '':
        return
    nc_dest = destination['neuroncores_in_use']
    nc_src = source['neuroncores_in_use']
    for nc_idx in nc_src:
        nc_util_src = nc_src[nc_idx]['neuroncore_utilization']
        flops_src = float(nc_src[nc_idx]['effective_flops'])
        if nc_idx in nc_dest:
            nc_dest[nc_idx]['neuroncore_utilization'] = min(
                100, nc_dest[nc_idx]['neuroncore_utilization'] + nc_util_src)
            nc_dest[nc_idx]['effective_flops'] += flops_src
        else:
            nc_dest[nc_idx] = {
                'neuroncore_utilization': nc_util_src,
                'effective_flops': flops_src
            }


def _init_aggregated_memory_used(destination):
    destination['error'] = ''
    destination['neuron_runtime_used_bytes'] = {
        'host': 0,
        'neuron_device': 0,
        'usage_breakdown': {
            'host': dict([(key, 0) for key in Const.MEM_USAGE_FIELD_NAMES[0]]),
            'neuroncore_memory_usage': {}
        }
    }
    destination['loaded_models'] = []


def _aggregate_memory_used(destination, source):
    if source['error'] != '':
        return
    used_src = source['neuron_runtime_used_bytes']
    used_dest = destination['neuron_runtime_used_bytes']
    for metric, data in used_src.items():
        if metric != 'usage_breakdown':
            used_dest[metric] += data
        else:
            if metric not in used_src:
                continue
            simple_aggr_objects = ['host']
            usage_breakdown_dest = used_dest['usage_breakdown']
            for aggr_name in simple_aggr_objects:
                bd_src = data[aggr_name]
                bd_dest = usage_breakdown_dest[aggr_name]
                for src_metric, src_value in bd_src.items():
                    if src_metric not in bd_dest:
                        bd_dest[src_metric] = 0
                    bd_dest[src_metric] += src_value

            if 'neuroncore_memory_usage' not in data:
                continue

            bd_src = data['neuroncore_memory_usage']
            bd_dest = used_dest['usage_breakdown']['neuroncore_memory_usage']
            for device, device_data in bd_src.items():
                if device not in bd_dest:
                    bd_dest[device] = {key:0 for key in Const.MEM_USAGE_FIELD_NAMES[1]}
                for field in device_data:
                    if field not in bd_dest[device]:
                        continue
                    bd_dest[device][field] += device_data[field]

    destination['loaded_models'] += source['loaded_models'][:]


def _init_aggregated_neuron_runtime_vcpu_usage(destination):
    destination['error'] = ''
    destination['vcpu_usage'] = {
        'user': 0,
        'system': 0
    }


def _aggregate_neuron_runtime_vcpu_usage(destination, source):
    if source['error'] != '':
        return
    vcpu_usage_src = source['vcpu_usage']
    vcpu_usage_dest = destination['vcpu_usage']
    for metric, value in vcpu_usage_src.items():
        vcpu_usage_dest[metric] += value


def _aggregate_current_runtimes(current_runtimes: dict) -> dict:
    aggregated = {}
    aggregated['report'] = {}
    aggregated['error'] = ''
    aggregated['neuron_runtime_tag'] = 'all'
    report = aggregated['report']
    
    this = globals()
    for metric in Const.METRIC_GROUP_NAMES:
        if metric not in report:
            report[metric] = {}
        handler_name = '_init_aggregated_' + metric
        handler_method = this[handler_name]
        handler_method(report[metric])

    for _, runtime_data in current_runtimes.items():
        current_report = runtime_data['report']
        for metric, metric_data in current_report.items():
            handler_name = '_aggregate_' + metric
            handler_method = this[handler_name]
            handler_method(report[metric], metric_data)

    return aggregated


def _process_monitor_data(monitor_data: dict):
    current_runtimes = {}
    if monitor_data and monitor_data['neuron_runtime_data']:
        for item in monitor_data['neuron_runtime_data']:
            if item['error'] == '':
                current_runtimes[item['pid']] = item

    current_runtimes[0] = _aggregate_current_runtimes(current_runtimes)
    return current_runtimes


def _log_monitor_data(writer: SummaryWriter, monitor_data: dict, timestamp: float):
    current_runtimes = _process_monitor_data(monitor_data)
    system_runtime = current_runtimes[0]

    total_tflops = 0.0
    for (nc, metrics) in system_runtime['report']['neuroncore_counters']['neuroncores_in_use'].items():
        writer.add_scalar('neuroncore_utilization/neuroncore_' + str(nc),
                          metrics['neuroncore_utilization'], timestamp)

        total_tflops += metrics['effective_flops']
        tflops = float(Utils.format_flops(metrics['effective_flops']).strip()[:-2])
        writer.add_scalar('effective_tflops/neuroncore_' + str(nc), tflops, timestamp)
    
    writer.add_scalar('effective_tflops_total', float(Utils.format_flops(total_tflops).strip()[:-2]), timestamp)
    average_tflops = total_tflops / max(1, len(system_runtime['report']['neuroncore_counters']['neuroncores_in_use']))
    average_tflops = float(Utils.format_flops(average_tflops).strip()[:-2])
    writer.add_scalar('effective_tflops_average', average_tflops, timestamp)

    gb = 1 << 30
    total_memory_used = 0.0
    for (nc, metrics) in system_runtime['report']['memory_used']['neuron_runtime_used_bytes']['usage_breakdown']['neuroncore_memory_usage'].items():
        total_of_all_fields = sum([metrics[field] for field in Const.MEM_USAGE_FIELD_NAMES[1]]) / gb
        total_memory_used += total_of_all_fields
        writer.add_scalar(f'memory_usage/neuroncore_{nc}', total_of_all_fields, timestamp)
        for field in Const.MEM_USAGE_FIELD_NAMES[1]:
            writer.add_scalar(f'memory_breakdown/neuroncore_{nc}/{field}', metrics[field] / gb, timestamp)
            
    writer.add_scalar('memory_usage_total', total_memory_used, timestamp)
    writer.add_scalar('memory_usage_average', total_memory_used / max(1, len(system_runtime['report']['memory_used']['neuron_runtime_used_bytes']['usage_breakdown']['neuroncore_memory_usage'])), timestamp)

    writer.flush()


def _main(args: argparse.Namespace):
    log_dir = Path(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    running = True
    def _handle_sigint(signal, frame):
        nonlocal running
        running = False
        writer.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    start_timestamp = time.time()
    for line in sys.stdin:
        if not running:
            break

        current_timestamp = time.time()
        monitor_data = json.loads(line)
        _log_monitor_data(writer, monitor_data, current_timestamp - start_timestamp)


def _parse_args():
    parser = argparse.ArgumentParser(description='Neuron Monitor Top Utility')
    parser.add_argument('log_dir', type=Path,
                        help='Directory to store neuron-monitor logs')
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    _main(args)
    