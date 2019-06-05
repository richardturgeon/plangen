
import glob
import itertools  as it
import json
import numpy as np
import os
import planargs
import sys

from abc import ABC, abstractmethod     # abstract class support
from scipy.special import comb
from pprint import pprint as pp

def isempty(path):
    """Determine whether the given directory is empty."""
    flist = glob.glob(os.path.join(path,'*'))
    return flist == []


def validate_args(args):
    """Validate the execution arguments as defined in planargs.py.

    This function validates input arguments defined in the 'args' namespace.
    The inputs are lists series of feature-set names (fs_names), files
    (fs_paths) and partitioning attributes (fs_parts). fs_names and fs_files
    must designate the same number of parameters. For example:

        --fs_names CELL DRUG --fs_paths cells.txt drugs.txt

    The CELL name is paired with the cells.txt file, DRUG with drugs.txt, etc.
    Currently, this one for one correspondence also applies to the fs_part arg,
    which specifies the number of partitions the feature-set list is broken
    into at every level of the plan generation recursion. A complete example
    might look like this:

        --fsnames CELL DRUG --fs_paths cells.txt drugs.txt --fs_parts 2 2

    An output directory for the plan in any of its formats is given by out_dir.
    An input directory may be specified via in_dir to simplify the coding of
    fs_paths. Otherwise, feature-set files must be fully specified. Each of the
    files is read and returned.

    DOCUMENT ALL ARGS ??????????????????????

    Returns:
        Upon success, a list of feature-set entry lists is returned. All entries
        are stripped of white-space, all white-space lines have been removed.
        For example:

            [[CELL1 ... CELLn] [DRUG1 ... DRUGn]]

        Additionally, args.generator is instantiated with a class defining the selected partition()
        function (future).
    """
    params = {}
    verbose = args.verbose

    fs_names_len = len(args.fs_names)
    fs_paths_len = len(args.fs_paths)
    fs_parts_len = len(args.fs_parts)

    nbr_feature_sets = fs_names_len
    test_lengths = [fs_names_len, fs_paths_len, fs_parts_len]
    reqd_lengths = [nbr_feature_sets] * 3

    if test_lengths != reqd_lengths:
        sys.exit("Error: The lengths of all feature set definition args (fs_<>) must be identical")

    if nbr_feature_sets <= 1:
        sys.exit("Error: Partitioning requires multiple feature sets")

    for nparts in args.fs_parts:
        if nparts <= 1 or nparts >= 8:
            sys.exit("Error: Invalid partitioning value %d" % nparts)

    # validate input and output directories
    if args.in_dir and not os.path.isdir(args.in_dir):
        sys.exit("Error: --in_dir must designate a directory, '%s' is not valid" % args.in_dir)

    if not os.path.isdir(args.out_dir):
        sys.exit("Error: --out_dir must designate a directory, '%s' is not valid" % args.out_dir)

    if not args.overwrite and not isempty(args.out_dir):
        sys.exit("Error: --out_dir '%s' is not empty, --overwrite not specified" % args.out_dir)

    if verbose:
        print("Writing plan information to %s" % os.path.abspath(args.out_dir))

    # expand, validate and load input feature-set content lists 
    feature_set_content = []
    file_error = False
    if args.in_dir == None:
        args.in_dir = ''    # prepare for use in os.path.join()

    for i, path in enumerate(args.fs_paths):
        fullpath = os.path.join(args.in_dir, path)
        if not os.path.exists(fullpath):
            file_error = True
            print("Error: %s file not found" % fullpath)
        else:
            with open(fullpath, 'r') as f:          # read text and sanitize
                raw_lines = f.readlines()

            lines = [line.strip() for line in raw_lines]
            lines = [l for l in lines if l != '']
            feature_set_content.append(lines)

            if verbose:
                print("Loading '%s' feature set definition from %s - %d lines"
                    % (args.fs_names[i], fullpath, len(lines)))

    if file_error:
        sys.exit("Terminating due to error")

    # construct a partitioning object exporting a partion() function

    if args.partition_strategy == 'windows':
        args.generator = WindowsSubsetGenerator()

    # return feature-set contents lists
    return feature_set_content


class SubsetGenerator(ABC):
    """Abstract class implementing a data partitioning method.

    The SubsetGenerator class provides a template for subclasses that implement
    mechanisms for dividing sets of lists into sublists for the purpose of 
    defining unique ML training and validation sets.

    Subclasses must implement those methods defined as @abstractmethod.
    The validate() function provided here does a sanity test for all anticipated
    partitioning schemes. Subclasses should implement their specializations.
    """

    def __init__(self, name=''):
        self.name = name
        self.term_msg = "Terminating due to error"

    @abstractmethod
    def partition(
        self,
        base,
        size=None,
        count=None,
        name='-unspecified-'
    ):
        """Partion a feature-set array.

        Partition the 'base', a list of elements, using the abstract arguments
        'size' and 'count' to tailor the implementation's algorithm. 'name' is
        used in error reporting and is optional.
        """
        validate(self, base, size, count, name)
        return []


    def _validation_error(self, base_len, size, count, name='-unspecified-'):
        """Provide a common error reporting function. """
        print("Base list length: %d requested %d sublists of length %d" %
            (base_len, count, size))


    def validate(self, base, size=None, count=None, name='-unspecified-'):
        """Provide basic request validation, specific generators may impose
        additional requirements.
        """
        berror = False
        base_len = len(base)

        if size == None or size <= 0 or size > base_len:
            berror = True
        else:
            unique_combos = comb(base_len, size)        # implements N take K
            if count > unique_combos:
                berror = True
        if berror:
            SubsetGenerator._validation_error(self, base_len, size, count, name)

        return not berror

#
# UNDER EVALUATION ?????????????????????????????????????????????????????
#

class IterativeSubsetGenerator(SubsetGenerator):
    """ Tom Brettin method... subset generation via iteration over base"""
    def __init__(self):
        SubsetGenerator.__init__(self, 'IterativeSubsetGenerator')

    def partition(self, base, size=None, count=0, name=None):
        """ """

        if size is None:
            print("Error: Unspecified list partitioning size")
            sys.exit(3)

        """
        base_len = len(base)
        if count == 0:              # a simplification useful in the iterative approach
            count = base_len
        """

        is_valid = SubsetGenerator.validate(self, base, size, count, name)
        if not is_valid:
            print(self.term_msg)
            sys.exit(1)

        if count > base_len:
            SubsetGenerator._validation_error(self, base_len, size, count, name)
            print(self.term_msg)
            sys.exit(2)

        np_base = np.array(base)
        selected_sublists = []
        omit_size = base_len - size
        increment = min(size, omit_size)

        # omit consecutive blocks of feature-name entries 
        for i in range(count):
            org = i * increment
            if org >= base_len:
                org = org % base_len
            if org == 0 and i > 0:
                print("Warning: %d sublists of %s completed short of the requested %d"
                    % (i, name, count))
                break

            end = org + size
            sublist = np_base.take(range(org, end), mode='wrap')
            print(sublist)
            selected_sublists.append(sublist)

        return selected_sublists

#
# RICK STEVEN'S PARTITIONING APPROACH 
#

class WindowsSubsetGenerator(SubsetGenerator):
    """Not-necessarily-square partitioning approach.

    DOCUMENTATION REQUIRED ?????????????
    """

    def __init__(self):
        SubsetGenerator.__init__(self, 'WindowsSubsetGenerator')

    def partition(self, base, size='n/a', count=1, name=None):
        """ partition list return sublists"""
        base_len = len(base)
        if base_len < count:            # can partition any further?
            return [base]               # document this ???????????????????????? 

        size = int((base_len / count) + .9)
        sublists = []

        for i in range(count):
            org = i * size
            end = org + size
            part = base[org:end]
            sublists.append(part)

        return sublists


def breakout(seq_list, names):
    dict = {}
    for seq, tag in zip(seq_list, names):
        dict[tag] = list(seq)
    return dict


def build_plan_tree(args, feature_set_content, subplan_id='', depth=0, data_pfx='', plan_pfx=''):
    """ recursive plan generation"""
    curr_depth = depth + 1
    all_parts = []
    successful_splits = 0

    #flat_partitions = []
    #files = []
    #sequence = 0

    for i in range(len(args.fs_names)):
        group = feature_set_content[i]
        count = args.fs_parts[i]
        feature_set_name = args.fs_names[i]
        partitions = args.generator.partition(feature_set_content[i], count=count)   # name= ??????????????
        
        if args.debug:
            pp(partitions)
      
        """
        if len(partitions) <= 1:
            return 0    #??????????????????????
        """
        if len(partitions) > 1:                     # partitioning successful?
            successful_splits += 1                  # successful split

        all_parts.append(partitions)

    # if no further partitioning is possible task is complete
    if successful_splits == 0:
        return 0

    # acquire a cross-product of all feature-set partitions
    parts_xprod = np.array(list(it.product(*all_parts)))
    steps = len(parts_xprod)
    substeps = 0

    for plan_id in range(steps):
        train = []
        val = []

        # split into validation and training components
        for i, plan in enumerate(parts_xprod):
            section = breakout(plan, args.fs_names)
            if i == plan_id:
                val.append(section)
            else:
                train.append(section)

        # generate next depth/level (successor) plans 
        new_subplan_id = '{}.{}'.format(subplan_id, plan_id + 1)
        args.plan_dict[new_subplan_id] = {'val': val, 'train': train}
        data_name = '{}.{}'.format(data_pfx, plan_id + 1)
        plan_name = '{}.{}'.format(plan_pfx, plan_id + 1)

        substeps += build_plan_tree(
            args,
            parts_xprod[plan_id],
            subplan_id=new_subplan_id,
            depth=curr_depth,
            data_pfx=data_name,
            plan_pfx=plan_name
        )

    steps += substeps
    return steps

    """
    # THIS IS A WORK-IN-PROGRESS ... GENERATING FILES FOR DATA AND PLAN

        files.append([])
        files_ndx = len(files) - 1

        for j in range(len(partitions)):
            part = partitions[j]
            flat_partitions.append(part)
            if len(part) == 0:
                sys.exit("big trouble ?????????????")

            sequence += 1
            file_name = '{}.{}.{}'.format(data_pfx, sequence, feature_set_name)
            print("writing file %s with %d entries" % (file_name, len(part)))  # write out 'part'
            #write_file(file_name, part)
            pair = (feature_set_name, file_name)
            files[files_ndx].append(pair)

    file_xprod = np.array(list(it.product(*files)))
    nbr_plans = len(file_xprod)

    for seq in range(nbr_plans):
        plan_string = ''

        for ndx, curr in enumerate(file_xprod):
            if ndx == seq:
                plan_string += '--val ('
            else:
                plan_string += '--inc ('
            for (tag, fname) in curr:
                plan_string += '{}-{} '.format(tag, fname)
            plan_string += ')'

        file_name = '{}.{}'.format(plan_pfx, seq + 1)
        print(file_name)
        plan_lines = list(plan_string)
        #write_file(file_name, plan_lines)

    # construct list of omitted feature entries

    for seq in range(nbr_plans):
        omitted_feature_content = []
        org = 0

        for i in partition_spec:
            omitted_feature_content.append(flat_partitions[org])
            org = i

        data_name = '{}.{}'.format(data_pfx, seq + 1)
        plan_name = '{}.{}'.format(plan_pfx, seq + 1)

        steps = build_plan_tree(
            args,
            omitted_feature_content,
            subplan_id=new_subplan_id,
            depth=curr_depth,
            data_pfx=data_name,
            plan_pfx=plan_name
        )
    return
    """

def write_file(fname, title, string_list):
    """ write text expressed as an array of lines to file """
    with open(fname, 'w') as f:
        for line in string_list:
            f.write(line)

def write_dict_to_json(dictionary, fname):
    """ write dictionary to a json file """
    with open(fname, 'w') as f:
        json.dump(dictionary, f)

#----------------------------------------------------------------------------------
# various hard-coded lists, test cases - the synthetic feature-sets remain useful
#----------------------------------------------------------------------------------

""" 
synthetic_cell_names = ['cell_' + '%04d' % (x) for x in range(4)]
synthetic_drug_names = ['drug_' + '%04d' % (x) for x in range(4)]

cell_names = [
    'NCI60.A549',
    'NCI60.MOLT-4',
    'NCI60.SW-620',
    'NCI60.OVCAR-8',
    'NCI60.U251',
    'NCI60.NCI-H23',
    'NCI60.SN12C',
    'NCI60.NCI-H460'
]

drug_names = [
    'NSC.740',
    'NSC.3053',
    'NSC.49842',
    'NSC.125066',
    'NSC.752',
    'NSC.26980',
    'NSC.63878',
    'NSC.82151'
]

generator = WindowsSubsetGenerator()

cell_sublists = generator.partition(cell_names, count=2, name='Cells')
print(cell_sublists)
drug_sublists = generator.partition(drug_names, count=3, name='Drug')
print(drug_sublists)
"""

#----------------------------------------------------------------------------------
# mainline 
#----------------------------------------------------------------------------------

# Acquire and validate arguments
args = planargs.parse_arguments()
feature_set_content = validate_args(args)        # returns a list of feature-set lists

# feature_set_content = [cell_names, drug_names] 
# feature_set_content = [synthetic_cell_names, synthetic_drug_names]

# Plan generation 
data_fname_pfx = os.path.join(args.out_dir, 'DATA.1')
plan_fname_pfx = os.path.join(args.out_dir, 'PLAN.1')

args.json = True                                # the only available option thus far
args.plan_dict = {}

steps = build_plan_tree(
    args,                                       # command linee argument namespace
    feature_set_content,                        # for example [[cell1 ... celln] [drug1 ... drugn]]
    subplan_id='1',                             # name of root plan, subplan names created from this stem
    data_pfx=data_fname_pfx,                    # DATA file prefix, building block for feature name files
    plan_pfx=plan_fname_pfx                     # PLAN file prefix, building block for plan name files
)

print("Plan generation complete, total steps: %d" %  steps)

if args.json:
    json_file_name = os.path.join(args.out_dir, 'plangen.json')
    json_abspath = os.path.abspath(json_file_name)
    write_dict_to_json(args.plan_dict, json_abspath) 
    print("%s JSON file written" % json_abspath)

if args.debug:
    pp(args.plan_dict, width=160)

