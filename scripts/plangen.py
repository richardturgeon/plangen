
import os
import sys
import json
import numpy as np
import itertools  as it
import planargs

from abc import ABC, abstractmethod     # abstract class support
from scipy.special import comb
from pprint import pprint as pp



class SubsetGenerator(ABC):
    """ """
 
    def __init__(self, working_dir, name=''):
        self.name = name
        self.working_dir = working_dir
        self.term_msg = "Terminating due to error"

    @abstractmethod
    def partition(
        self,
        base,
        size=None,
        count=None,
        name='-unspecified-'
    ):
        validate(self, base, size, count, name)
        return 0


    def _validation_error(self, base_len, size, count, name='-unspecified-'):
        """ """
        print("Base list length: %d requested %d sublists of length %d" %
            (base_len, count, size))


    def validate(self, base, size=None, count=None, name='-unspecified-'):
        """ basic request validation, specific generators may impose
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

class IterativeSubsetGenerator(SubsetGenerator):
    """ subset generation via iteration over base """

    def __init__(self, working_dir):
        SubsetGenerator.__init__(self, working_dir, 'IterativeSubsetGenerator')

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

class QuadrantSubsetGenerator(SubsetGenerator):
    """ not-necessarily-square partitioning approach """

    def __init__(self, working_dir):
        SubsetGenerator.__init__(self, working_dir, 'QuadrantSubsetGenerator')

    def partition(self, base, size='n/a', count=1, name=None):
        base_len = len(base)
        if base_len < count:
            return []

        size = int((base_len / count) + .9)
        sublists = []

        for i in range(count):
            org = i * size
            end = org + size
            part = base[org:end]
            sublists.append(part)

        return sublists

class RandomSubsetGenerator(SubsetGenerator):
    """ subset generation via random selection from base """
    pass


#
#
#

def get_feature_set_info(working_dir):
    """ """
    print("intro ... ")

    feature_set_names = []
    feature_set_paths = []

    while True:
        print("How many feature sets?")
        response = sys.stdin.readline()
        response = response.strip()
        if response == '' or response == '0':
            nbr_feature_sets = 0
            break
        try:
            nbr_feature_sets = int(response)
        except ValueError:
            continue
        break

    for i in range(1, nbr_feature_sets + 1):
        while True:
            print("Enter the feature set %d name and its file path" % i)
            response = sys.stdin.readline()
            tokens = response.split()
            if len(tokens) != 2:
                continue

            name = tokens[0].upper()
            path = tokens[1]

            if not os.path.isabs(path):
                path = os.path.join(working_dir, path)
            if not os.path.isfile(path):
                print("File not found")
                continue
            if name in feature_set_names:
                print("The %s feature set has already been defined")
                continue

            feature_set_names.append(name)
            feature_set_paths.append(path)
            break

    return nbr_feature_sets, feature_set_names, feature_set_paths


def get_working_dir():
    while True:
        print("Specify the working directory or <enter> for the current directory")
        response = sys.stdin.readline()
        response = response.strip()
        if response == '':
            response = './'
            break

        if not os.path.isdir(response):
            print("%s is not a directory" % respose)
            continue

    return response

def breakout(seq_list, names):
    dict = {}
    for seq, tag in zip(seq_list, names):
        dict[tag] = list(seq)
    return dict


def build_tree(generator, params, feature_set_content, plan_dict, plan_key=None, depth=0, data_pfx='', plan_pfx=''):
    this_depth = depth + 1
    partition_spec = params['partition_spec']
    feature_set_names = params['feature_set_names']
    verbose = params['verbose']

    flat_partitions = []
    files = []
    sequence = 0

    all_parts = []
    for i in range(len(feature_set_content)):
        group = feature_set_content[i]
        count = partition_spec[i]
        feature_set_name = feature_set_names[i]
        partitions = generator.partition(feature_set_content[i], count=count)   # name= ??????????????
#       print(partitions)

        if len(partitions) == 0:
            return # ??????????????????????????????????????? fix this per M 

        all_parts.append(partitions)

    parts_xprod = np.array(list(it.product(*all_parts)))
    local_steps = len(parts_xprod)
    params['total_steps'] += local_steps

    for plan_id in range(local_steps):
        train = []
        for i, plan in enumerate(parts_xprod):

            # define validations 

            if i == plan_id:
                val = [breakout(plan, feature_set_names)]
            else:
                train.append(breakout(plan, feature_set_names))

        new_plan_key = '{}.{}'.format(plan_key, plan_id + 1)
        plan_dict[new_plan_key] = {'val': val, 'train': train}

        data_name = '{}.{}'.format(data_pfx, plan_id + 1)
        plan_name = '{}.{}'.format(plan_pfx, plan_id + 1)
        build_tree(generator, params, parts_xprod[plan_id], plan_dict, plan_key=new_plan_key, depth=this_depth, data_pfx=data_name, plan_pfx=plan_name)


        """
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
        """
    """
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
        build_tree(generator, params, omitted_feature_content, plan_dict, plan_key=new_plan_key, depth=this_depth, data_pfx=data_name, plan_pfx=plan_name)
    """
    return

def write_file(fname, title, string_list):
    """ write text expressed as an array of lines to file """
    with open(fname, 'w') as f:
        for line in string_list:
            f.write(line)

def write_dict_to_json(dictionary, fname): 
    """ write dictionary to a json file """
    with open(fname, 'w') as f:
        json.dump(dictionary, f)
    with open(fname, 'r') as f:
        new_dict = json.load(f)

#----------------------------------------------------------------------------------
# MAINLINE
#----------------------------------------------------------------------------------

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


#working_dir = get_working_dir()
working_dir = './'
generator = QuadrantSubsetGenerator(working_dir)

# sublist test -------------
"""
cell_sublists = generator.partition(cell_names, count=2, name='Cells')
print(cell_sublists)
drug_sublists = generator.partition(drug_names, count=3, name='Drug')
print(drug_sublists)
"""
# sublist test -------------


#nbr_feature_sets, feature_set_names, feature_set_paths = get_feature_set_info(working_dir)

args = planargs.parse_arguments()

nbr_feature_sets = 2
feature_set_names = ['CELL', 'DRUG']
partition_spec = [2, 2]
feature_set_paths = []

params = {}
params['working_dir'] = working_dir
params['nbr_feature_sets'] = nbr_feature_sets
params['feature_set_names'] = feature_set_names
params['feature_set_paths'] = feature_set_paths
params['partition_spec'] = partition_spec
params['total_steps'] = 0
params['verbose'] = True

feature_set_content = [cell_names, drug_names]          # synthetic 
plan_dict = {}

build_tree(generator, params, feature_set_content, plan_dict, plan_key='1', data_pfx='~/treetest/DATA.1', plan_pfx='~/treetest/PLAN.1')

print("Plan complete, total steps: %d" % params['total_steps'])
pp(plan_dict, width=160)
write_dict_to_json(plan_dict, 'dq_plan.json')
write_file('json_plan', 'JSON plan file', [jstring])


