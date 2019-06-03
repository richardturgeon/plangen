"""
plangen command line argument definitions
"""

import os
import sys
import glob
import argparse

partitioning_strategies = ['exclude_one', 'test2', 'test3']

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_dir',
                        type=str,
                        default='inputs',
                        help='Directory containing feature-set list files')

    parser.add_argument('--out_dir',
                        default='results',
                        type=str,
                        help='Directory to contain generated plan files')

    parser.add_argument('--overwrite',
                        default=False,
                        action='store_true',
                        help='Accept non-empty out_dir, contents overwritten')

    parser.add_argument('--verbose',
                        default=False,
                        action='store_true',
                        help='Trace execution')

    parser.add_argument ('--partition_strategy',
                        choices=partitioning_strategies,
                        default=partitioning_strategies[0],
                        help='Specify a feature-set partitioning strategy')

    # The following fs_* arguments are required, the number of values specified for each
    # must match, and at least two values are required for each 

    parser.add_argument('--fs_names',
                        required=True,
                        type=str,
                        nargs='+',
                        help='Specify a list of (arbitrary) feature-set names')

    parser.add_argument('--fs_parts',
                        required=True,
                        type=str,
                        nargs='+',
                        help='Specify a list of partition counts')

    parser.add_argument('--fs_files',
                        required=True,
                        type=str,
                        nargs='+',
                        help='Specify a list of partition counts')

    args= parser.parse_args()
    return args

