
import glob
import itertools as it
import json
import numpy as np
import os
import planargs
import sys

from abc import ABC, abstractmethod     # abstract class support
from collections import OrderedDict
from scipy.special import comb
from pprint import pprint as pp
from datetime import datetime

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

    Returns:
        Upon success, a tuple is returned. It contains:

            t[0] - the generator class implementing the appropriate partition()
                   function.

            t[1] - a list of feature-set entry lists is returned. All entries
                   are stripped of white-space, all white-space lines have been removed.
                   For example:

                        [[CELL1 ... CELLn] [DRUG1 ... DRUGn]]

        Additionally, an args.lines list is created where each entry contains
        the entry count of the corresponding fs_paths file argument.
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
    fs_content = []
    args.fs_lines = []
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

            text = [line.strip() for line in raw_lines]
            text = [l for l in text if l != '']
            fs_content.append(text)
            args.fs_lines.append(len(text))

            if verbose:
                print("Loading '%s' feature set definition from %s - %d lines"
                    % (args.fs_names[i], fullpath, len(text)))

    if file_error:
        sys.exit("Terminating due to error")

    # construct a partitioning object exporting a partion() function
    if args.partition_strategy == 'windows':
        generator = WindowsSubsetGenerator()
    # return feature-set contents lists
    return generator, fs_content


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

    def get_plan_label(self, plan_dict, root_name):
        root = plan_dict[root_name]
        return root['label']

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


class WindowsSubsetGenerator(SubsetGenerator):
    """CANDLE milestone 13 style feature set partitioning.

    All SubsetGenerator subclasses are required to implement partition(),
    plan_init() and plan_term() functions.
    """

    def __init__(self):
        SubsetGenerator.__init__(self, 'WindowsSubsetGenerator')
        self.strategy = "windows"

    def plan_init(self, fs_names, fs_paths, fs_lines, fs_parts, root_name='1'):
        """Initialize - collect plan metadata """
        currtime = datetime.now()
        details = {'fs_names': fs_names, 'fs_filepaths':fs_paths, 'fs_parts': fs_parts}
        details['create_date'] = currtime.strftime("%m/%d/%Y-%H:%M:S")
        details['strategy'] = self.strategy

        label = ''
        for i in range(len(fs_names)):
            if i != 0:
                label += '.'
            s = '{}{}-p{}'.format(fs_names[i], fs_lines[i], fs_parts[i])
            label += s

        details['label'] = label
        plan_dict = OrderedDict()
        plan_dict[root_name] = details
        return root_name, plan_dict

    def plan_term(self, plan_dict, root_name, nbr_nodes):
        """Completion - post plan summary  metadata """
        meta = plan_dict[root_name]
        meta['nodes'] = nbr_nodes


    def partition(self, base, size='n/a', count=None, name=None):
        """Partition a feature-set list into lists of equal sized elements.

        This partitioner accepts a list of feature-set names and returns
        'count' lists, the elements evenly divided between these lists.
        The last sublist will contain fewer elements if the base list cannot
        be evenly divided.

        Args:
            base:   A list of feature-set names.
            size:   Ignored, not used in this implementation.
            count:  The number of equal sized partitions requested.
                    Required, the minimum value is 2.
            name:   A tag used for debug/error tracing. Not used in this
                    implementation.

            These arguments are common to all partition functions defined in
            SubsetGenerator subclasses.

        Returns:
            When the input 'base' list contains a number of entries equal to or
            greater than 'count', a list of 'count' sublists is returned. For
            example:

                [[CELL1, ..., CELL4], [CELL5, ..., CELL7]]

            Otherwise, the given 'base' list is returned as a list of one list.
        """

        base_len = len(base)
        if base_len < count:            # can partition any further?
            return [base]               # return the input as a list of one list

        size = int((base_len / count) + .9)
        sublists = []

        for i in range(count):
            org = i * size
            end = org + size
            part = base[org:end]
            sublists.append(part)

        return sublists

#------------------------------------------------------------------------------
# Plan navigation, content retrieval 
#------------------------------------------------------------------------------

def load_plan(filepath):
    """Load a JSON transfer learning plan.

    The named JSON tranfer learning plan file is loaded in a manner that preserves
    the entry order imposed when the plan was created. This allows the root entry
    to be easily located regardless of the plan entry naming scheme in use.

    Args:
        filepath:   A relative or absolute path to the JSON file.

    Returns:
        An entry-ordered plan in OrderedDict format is returned.
    """

    with open(filepath, 'r') as f:
        ordered_plan_dict = json.load(f, object_pairs_hook=OrderedDict)
    return ordered_plan_dict


def get_node(plan_dict, node_name=None):
    """Retrieve the content of a named plan node or the root node.

    Args:
        plan_dict:      The plan dictionary as returned by load_plan().
        node_name:      The name of the desired node. Omit this arg to acquire
                        the content and name of the plan tree root node.

    Returns:
        A (content, node_name) pair is returned. The returned name is useful when
        using default arguments to retrieve the root node.
    """

    if node_name is None:
        node_name, content = next(iter(plan_dict.items()))
    else:
        content = plan_dict.get(node_name)
    return content, node_name


def get_predecessor(plan_dict, node_name):
    """Acquire the name of the predecessor (parent) of a given node.

    The plan tree is a true tree. All nodes except for the root have
    exactly one predecessor / parent. Use this function to walk 'up'
    the tree.

    Args:
        plan_dict:      The plan dictionary as returned by load_plan().
        node_name:      The name of the target node (often the current node).

    Returns:
        The name of the parent node is returned. If the root node_name is
        specified None is returned.
    """

    segments = node_name.split(sep='.')
    if len(segments) <= 1:
        node_name = None
    else:
        segments.pop()
        node_name = '.'.join(segments)
    return node_name


def get_successors(plan_dict, node_name):
    """Acquire the names of the successors (children) of a given node.

    All nodes other than 'leaf' nodes have at least one successor. Use
    this function to walk 'down' one or more plan subtrees.

    Args:
        plan_dict:      The plan dictionary as returned by load_plan().
        node_name:      The name of the target node (often the current node).

    Returns:
        A list of the names of all successors (children) of the given node
        is returned. The list may be empty.
    """
    successor_names = []
    for i in it.count(start=1):
        new_name = node_name + '.' + str(i)
        value = plan_dict.get(new_name)
        if not value:
            break
        successor_names.append(new_name)

    return successor_names


def parse_plan_entry(plan_entry):
    """ THIS NEEDS REFINEMENT ?????????? """
    return plan_entry['train'], plan_entry['val']

#------------------------------------------------------------------------------
# Plan construction 
#------------------------------------------------------------------------------

def build_dictionary_from_lists(seq_list, names):
    """Create a dictionary with 'names' as labels and 'seq_list' values."""
    dict = {}
    for seq, tag in zip(seq_list, names):
        dict[tag] = list(seq)
    return dict


def build_plan_tree(args, feature_set_content, parent_plan_id='', depth=0, data_pfx='', plan_pfx=''):
    """Generate a plan supporting training, transfer-learning, resume-training.

    ADD GENERAL DOC

    This function is recursive.

    Arguments:
        args:       A namespace capturing the values of command line arguments
                    and parameter values derived from those arguments. Refer to
                    validate_args().

        feature_set_content: This is a list of sublists, where each sublist
                    contains the names of the nth group of feature-set elements.

        parent_plan_id: This is the name of the parent's plan. The name
                    is extended with '.nn' at each level of the recursion to
                    ensure that parentage/liniage is fully conveyed in each
                    (subplan) plan_id.

        depth:      Specify 0 on the root call. This arg can be used to
                    determine/set the current level of the recursion. It is
                    not currently used in the existing data partitioning /
                    file naming strategies.

        data_pfx:   Reserved for constructing feature-set name files.
        plan_pfx:   Reserved for constructing plan control files.

    Returns:
        args.plan_dict contains a dictionary representing the plan. This may be
        JSONized.

        The number of planning steps (nbr of nodes in the plan tree) is explicitly
        returned.
    """
    curr_depth = depth + 1
    all_parts = []
    successful_splits = 0

    #flat_partitions = []                           # preserve, used for file-based approach
    #files = []                                     # preserve, used for file-based approach     
    #sequence = 0                                   # preserve, used for file-based approach

    for i in range(len(args.fs_names)):
        group = feature_set_content[i]
        count = args.fs_parts[i]
        feature_set_name = args.fs_names[i]
        partitions = args.generator.partition(feature_set_content[i], count=count)   # name= ??????????????


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

    for step in range(steps):
        train = []
        val = []

        # split into validation and training components
        for i, plan in enumerate(parts_xprod):
            section = build_dictionary_from_lists(plan, args.fs_names)
            if i == step:
                val.append(section)
            else:
                train.append(section)

        # generate next depth/level (successor) plans 
        curr_plan_id = '{}.{}'.format(parent_plan_id, step + 1)
        args.plan_dict[curr_plan_id] = {'val': val, 'train': train}
        data_name = '{}.{}'.format(data_pfx, step + 1)
        plan_name = '{}.{}'.format(plan_pfx, step + 1)

        substeps += build_plan_tree(
            args,
            parts_xprod[step],
            parent_plan_id=curr_plan_id,
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
            parent_plan_id=curr_plan_id,
            depth=curr_depth,
            data_pfx=data_name,
            plan_pfx=plan_name
        )
    return
    """

def write_file(fname, title, string_list):
    """Write text expressed as an array of lines to file."""
    with open(fname, 'w') as f:
        for line in string_list:
            f.write(line)

def write_dict_to_json(dictionary, fname):
    """Write dictionary to a json file."""
    with open(fname, 'w') as f:
        json.dump(dictionary, f)

#----------------------------------------------------------------------------------
# various hard-coded lists, test cases - the synthetic feature-sets remain useful
#----------------------------------------------------------------------------------

"""
synthetic_cell_names = ['cell_' + '%04d' % (x) for x in range(1000)]
synthetic_drug_names = ['drug_' + '%04d' % (x) for x in range(1000)]
"""

#----------------------------------------------------------------------------------
# mainline 
#----------------------------------------------------------------------------------

def main():
    # Acquire and validate arguments
    args = planargs.parse_arguments()
    args.json = True                    # the only available option thus far

    generator, feature_set_content = validate_args(args)
    args.generator = generator

    root_name, args.plan_dict = generator.plan_init(
        fs_names=args.fs_names,         # validated cmdline arg
        fs_paths=args.fs_paths,         # validated cmdline arg
        fs_lines=args.fs_lines,         # created by validate_args
        fs_parts=args.fs_parts          # validated cmdline arg
    )

    # feature_set_content = [cell_names, drug_names] 
    # feature_set_content = [synthetic_cell_names, synthetic_drug_names]

    # Plan generation 
    data_fname_pfx = os.path.join(args.out_dir, 'DATA.1')
    plan_fname_pfx = os.path.join(args.out_dir, 'PLAN.1')

    steps = build_plan_tree(
        args,                           # command line argument namespace
        feature_set_content,            # for example [[cell1 ... celln] [drug1 ... drugn]]
        parent_plan_id=root_name,       # name of root plan, subplan names created from this stem
        data_pfx=data_fname_pfx,        # DATA file prefix, building block for feature name files
        plan_pfx=plan_fname_pfx         # PLAN file prefix, building block for plan name files
    )

    generator.plan_term(args.plan_dict, root_name, steps)
    print("Plan generation complete, total steps: %d" %  steps)

    if args.json:
        label = args.generator.get_plan_label(args.plan_dict, root_name)
        qualified_name = 'plangen.' + label + '.json'
        json_file_name = os.path.join(args.out_dir, qualified_name)
        json_abspath = os.path.abspath(json_file_name)
        write_dict_to_json(args.plan_dict, json_abspath)
        print("%s JSON file written" % json_abspath)

    if args.debug:
        pp(args.plan_dict, width=160)

    if args.test:
        test(json_abspath)

#----------------------------------------------------------------------------------
# test plan navigation and plan entry (node) retrieval
#----------------------------------------------------------------------------------

def test(plan_path):
    print("\nBegin plan navigation and node retrieval test")
    plan_dict = load_plan(plan_path)

    # plan root node name, value returned when node_name= is omitted
    metadata, root_name = get_node(plan_dict)

    # the root node has no parent / predecessor
    parent_name = get_predecessor(plan_dict, root_name)
    print("Demonstrate that root predecessir is not definede: %s" % parent_name)

    # the root node contains metadata, it is not a run specification
    successor_names = get_successors(plan_dict, root_name)
    print("\nThe first runable configurations are defined in %s\n" % successor_names)

    # the root node is the predecessor of these first level runables
    for sname in successor_names:
        parent_name = get_predecessor(plan_dict, sname)
        print("The parent of %s is %s" % (sname, parent_name))

    # run the right subtree
    print("\nRun the rightmost subtree \n")
    for i in it.count(start = 1):
        listlen = len(successor_names)
        if listlen == 0:
            break
        select_one  = successor_names[listlen - 1]
        parent_name = get_predecessor(plan_dict, select_one)
        print("%-16s is a successor of %-16s - all successors: %s" % (select_one, parent_name, successor_names))
        successor_names = get_successors(plan_dict, select_one)

        value,_  = get_node(plan_dict, select_one)
        train_set, validation_set = parse_plan_entry(value)

        # print the coarsely parsed plan entry train/validation components
        if i == 1:
            print("\n*** Training set ***")
            pp(train_set, width=160)
            print("\n*** Validation set ***")
            pp(validation_set, width=160)
            print(" ")

    print("\nEnd of branch reached")

#----------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

