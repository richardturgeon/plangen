
from collections import namedtuple
from enum import Enum
import glob
import itertools as it
import json
import numpy as np
import os
import sys
import sqlite3 
from sqlite3 import Error as db_Error

import planargs

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

    Returns
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
        generator = LeaveoutSubsetGenerator()
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


class LeaveoutSubsetGenerator(SubsetGenerator):
    """CANDLE milestone 13 style feature set partitioning.

    All SubsetGenerator subclasses are required to implement partition(),
    plan_init() and plan_term() functions.
    """

    def __init__(self):
        SubsetGenerator.__init__(self, 'LeaveoutSubsetGenerator')
        self.strategy = "leaveout"

    def plan_init(self, fs_names, fs_paths, fs_lines, fs_parts, root_name='1'):
        """Initialize - collect plan metadata """
        currtime = datetime.now()
        details = {'fs_names': fs_names, 'fs_filepaths':fs_paths, 'fs_parts': fs_parts}
        details['create_date'] = currtime.strftime("%m/%d/%Y-%H:%M:%S")
        details['strategy'] = self.strategy

        label = ''
        for i in range(len(fs_names)):
            if i != 0:
                label += '_'
            s = '{}{}-p{}'.format(fs_names[i], fs_lines[i], fs_parts[i])
            label += s

        details['label'] = label
        plan_dict = OrderedDict()
        plan_dict[root_name] = details
        return root_name, plan_dict

    def plan_term(self, plan_dict, root_name, nbr_subplans):
        """Completion - post plan summary  metadata """
        meta = plan_dict[root_name]
        meta['nbr_subplans'] = nbr_subplans


    def partition(self, base, size='n/a', count=None, name=None):
        """Partition a feature-set list into lists of equal sized elements.

        This partitioner accepts a list of feature-set names and returns
        'count' lists, the elements evenly divided between these lists.
        The last sublist will contain fewer elements if the base list cannot
        be evenly divided.

        Args
            base:   A list of feature-set names.
            size:   Ignored, not used in this implementation.
            count:  The number of equal sized partitions requested.
                    Required, the minimum value is 2.
            name:   A tag used for debug/error tracing. Not used in this
                    implementation.

            These arguments are common to all partition functions defined in
            SubsetGenerator subclasses.

        Returns
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
# Database support, table and column definitions, DDL and DML
# Refer to the plan_prep() function for a discussion of the "planstat" and
# "runhist" tables defined below.
#------------------------------------------------------------------------------

class RunType(Enum):
    RUN_ALL = 0
    RESTART = 1

class RunStat(Enum):        # subplan execution status
    SCHEDULED = 'scheduled'
    COMPLETE  = 'complete'

# planstat table, rows are returned via the PlanstatRow namedtuple

_planstat_ddl = """
    CREATE TABLE IF NOT EXISTS planstat (
        plan_name       TEXT NOT NULL PRIMARY KEY,
        create_date     TEXT NOT NULL,
        feature_sets    TEXT NOT NULL,
        partitions      TEXT NOT NULL,
        nbr_subplans    INTEGER
    ); """

PlanstatRow = namedtuple('PlanstatRow',
    [
        'rowid',
        'plan_name',
        'create_date',
        'feature_sets',
        'partitions',
        'nbr_subplans'
    ]
)

_select_row_from_planstat = """
    SELECT rowid, 
        plan_name, create_date, feature_sets, partitions, nbr_subplans
        FROM planstat
        WHERE plan_name='{}'
    """

_insert_planstat_plan = """
    INSERT INTO planstat (
        plan_name, create_date, feature_sets, partitions, nbr_subplans)
    VALUES ('{}', '{}', '{}', '{}', {})
    """

_delete_planstat_plan = """
    DELETE FROM planstat where rowid = {}
    """

# runhist table, rows are returned via the RunhistRow namedtuple

_runhist_ddl = """
    CREATE TABLE IF NOT EXISTS runhist (
        plan_id         INTEGER NOT NULL,
        subplan_id      TEXT NOT NULL,
        status          TEXT NOT NULL,
        start_time      TEXT NOT NULL,
        stop_time       TEXT,
        mae             REAL,
        mse             REAL,
        r_square        REAL,
        weights_fn      TEXT,
        PRIMARY KEY (plan_id, subplan_id)
    ); """

RunhistRow = namedtuple('RunhistRow',
    [
        'plan_id',
        'subplan_id',
        'status',
        'start_time',
        'stop_time',
        'mae',
        'mse',
        'r_square',
        'weights_fn'
    ]
)

_select_row_from_runhist = """
    SELECT plan_id, subplan_id, status, start_time, stop_time, mae, mse, r_square, weights_fn
    FROM runhist
    WHERE plan_id = {} and subplan_id = '{}'
    """

_insupd_scheduled_runhist = """
    REPLACE INTO runhist(plan_id, subplan_id, status, start_time,
        stop_time, mae, mse, r_square, weights_fn)
    VALUES({}, '{}', '{}', '{}',
        NULL, NULL, NULL, NULL, NULL)
    """

_insupd_completed_runhist = """
    UPDATE runhist SET
        status = '{}',
        stop_time = '{}',
        mae = {},
        mse = {},
        r_square = {},
        weights_fn = '{}'
    WHERE
        plan_id = {} AND subplan_id='{}'
     """

_delete_from_runhistory = """
    DELETE FROM runhist where plan_id = {}
    """

#------------------------------------------------------------------------------
# "Plan management" Database functions
#
#   db_connect          - establish database connection returning conn handle     
#   execute_sql_stmt    - execute a SQL statement with optional error trap   
#   plan_prep           - prepare for the execution of a multi-step "plan"
#   start_subplan       - start a subplan, (ex. '1.4.8'), write RunHistRow  
#   stop_subplan        - stop a subplan, update RunHistRow 
#   get_subplan_runhist - return a RunHistRow for a given subplan 
#   plan_remove         - remove all database records for the named plan
#------------------------------------------------------------------------------

def execute_sql_stmt(conn, stmt, cursor=None, trap_exception=False):
    """Execute a SQL statement.

    This is a convenience function that wraps the execution of a given SQL
    statement with exception handling and cleanup logic.

    Args
        conn:           An open database connection handle
        stmt:           A fully instantiated SQL statement

        cursor:         Optionally, a cursor managed by the caller. If
                        local cursor is used. Provide a cursor if you must
                        operate on it after completion, fetchall() for example.

        trap_exception: By default exceptions raised by the database must be
                        handled by the caller. If True, errors are reflected
                        by the boolean return value and the cursor and/or
                        connection handle provided by the caller are closed..

    Returns
        False indicates that an exception occurred, else True.
    """

    if cursor:
        lclcsr = cursor
    else:
        lclcsr = conn.cursor()
    try:
        db_exception = False
        lclcsr.execute(stmt)

    except db_Error as e:
        db_exception = True
        print('execute_sql_stmt:', stmt)
        print('execute_sql_stmt:', e)
        if not trap_exception:
            raise
    finally:
        if not cursor:
            lclcsr.close()

        if db_exception:
            if cursor:
                cursor.close()
            conn.close()

    return not db_exception


def db_connect(db_path):
    """Connect to the plan management database.

    Establish a connection to the sqlite3 database contained in the named file.
    A plan management database is created and populated at db_path if the file 
    does not exist.

    Args
        db_path:    A relative or absolute path or ":memory:"

    Returns
        A connection handle is returned upon success, else None
    """

    if db_path == ':memory:' or not os.path.exists(db_path):
        prev_allocated = False
    else:
        prev_allocated = True

    try:
        conn = sqlite3.connect(db_path)
    except db_Error as error:
        print('db_connect', error)
        raise

    # create plan management tables on initial database allocation 
    if conn and not prev_allocated:
        complete  = execute_sql_stmt(conn, _planstat_ddl)
        complete &= execute_sql_stmt(conn, _runhist_ddl)

        if complete:
            conn.commit()
        else:
            conn.close()
            conn = None
    return conn


def plan_remove(db_path, plan_path):
    """Delete the named plan from the plan managment database.

    The relative plan name is extracted from the plan_path by removing the
    leading directories and the trailing filetype suffix from the given
    plan_path. The planstat row is retrieved and the associated rowid is
    the plan_id identifying the target runhist table rows. 

    Returns
        Zero indicates deletion complete, -1 if the plan name is not matched.
    """

    status = 0
    conn = db_connect(db_path)
    plan_key = _get_planstat_key(plan_path)
    stmt = _select_row_from_planstat.format(plan_key)
    csr  = conn.cursor()
    execute_sql_stmt(conn, stmt, cursor=csr)
    nrow = csr.rowcount
    row  = csr.fetchone()

    print("%d run history rows deleted" % nrow)

    if not row:
        print("Error: CLEANUP request failed - %s has not been run" % plan_key)
        status = -1
    else:
        plan_rec = PlanstatRow._make(row)           # column-name addressable
        rowid = plan_rec.rowid                      # the unique rowid is the plan uniquifier
        _delete_runhistory(conn, rowid)
        stmt = _delete_planstat_plan.format(rowid)
        status = execute_sql_stmt(conn, stmt)

    csr.close()
    conn.close()
    return status


def plan_prep(db_path, plan_path, run_type=RunType.RUN_ALL):
    """Prepare to run a plan, a hierarchy of interdependent subplans.

    Plans names and related information are stored in the planstat (PLAN STATUS)
    table. There is one row for each plan submitted. A positive, unique integer
    called the 'rowid' is assigned to table rows by the database manager. The
    rowid of a planstat table row is defined here as the "plan_id". The plan_id
    together with a textual "subplan_id" (example: '1.2.4') form a composite
    key that is the primary key of the runhist (RUN HISTORY) table. The purpose
    of this function is to register the plan and return the associated plan_id.

    RunTypes
        When a new plan is presented it is registered in the planstat table and
        during its execution a large number of runhist (RUN HISTORY) table
        entries are created and then updated. To prevent unintended loss of
        data one of thethe following "RunTypes" is specified on the initial
        plan_prep() call and again on subsequent subplan_start() calls.

        Specify RUN_ALL on the first attempt to run a plan. If the plan name
        is already registered, the request fails and neither the planstat or
        runstat tables are changed.

        Specify RESTART if a prior attempt to run a plan did not complete. The
        presence of a corresponding planstat record is verified. subplan_start()
        returns a SKIP status if the associated runhist row (if any) is marked
        COMPLETE.

    Args
        db_path:        plan management database path (relative or absolute)
        plan_path:      JSON plan file (relative or absolute)
        run_type:       RunType.RUN_ALL, the default, or RunType.RESTART

    Returns
        A negative value indicates a fatal error.

        Otherwise the integer returned is the plan_id used together with a
        subplan_id string used in subsequent subplan_start(), subplan_stop()
        and get_subplan_hist() calls.
    """

    # load the plan and retrieve identity info
    plan_dict    = load_plan(plan_path)
    create_date  = get_plan_create_date(plan_dict)
    feature_sets = get_plan_fs_names(plan_dict)
    partitions   = get_plan_fs_parts(plan_dict)
    nbr_subplans    = get_plan_nbr_subplans(plan_dict)

    # de    termine if a plan of the given name has already been registered 
    conn = db_connect(db_path)
    plan_key = _get_planstat_key(plan_path)
    stmt = _select_row_from_planstat.format(plan_key)
    csr  = conn.cursor()
    execute_sql_stmt(conn, stmt, cursor=csr)
    row  = csr.fetchone()

    if not row:
        rowid = -1
    else:
        plan_rec = PlanstatRow._make(row)           # column-name addressable
        rowid = plan_rec.rowid                      # the unique rowid will be the uniquifier returned

    # compare run_type to initial expectations  
    error = False

    if run_type == RunType.RUN_ALL and rowid > 0:
        print("Error: RUN_ALL specified but plan: %s has already been defined" %  plan_key)
        error = True

    elif run_type == RunType.RESTART and rowid < 0:
        print("Error: RESTART specified but plan: %s has not been previously run" % plan_key)
        error = True

    elif rowid > 0 and create_date != create_date:  # DEBUG ???????????????????????????????????? plan_rec.create_date:
        print("Error: RESTART specified but the signature of the previously defined plan: %s does not match" % plan_key)
        error = True

    # register new plans acquiring the uniquifying plan_id used to compose runhistory table keys
    if not error and rowid < 0:
        feature_sets = str(feature_sets)
        feature_sets = feature_sets.replace("'", "") # create string literal from list of str
        partitions   = str(partitions)               # create string literal from list of int

        stmt = _insert_planstat_plan.format(
            plan_key,
            create_date,
            feature_sets,
            partitions,
            nbr_subplans
        )

        status = execute_sql_stmt(conn, stmt, cursor=csr)
        rowid = csr.lastrowid

    # cleanup resources and return uniquifier or error indicator    
    csr.close()
    conn.commit()

    if error:
        return -1
    else:
        return rowid


def start_subplan(db_path, plan_path, plan_id=None, subplan_id=None, run_type=None):
    """Schedule the execution of a subplan.

    This function writes a RunHistRow record to the runhist table indicating that
    the named plan/subplan has been SCHEDULED. The row includes the "start time".
    If the given run_type is RESTART, it is possible that the subplan has already
    run, as indicated by the status returned.

    Args
        db_path:        plan management database path (relative or absolute)
        plan_path:      JSON plan file (relative or absolute)
        plan_id:        the plan identifier returned by plan_prep()
        subplan_id      the subplan identifier ex. '1 4.8'
        run_type:       RunType.RUN_ALL or RunType.RESTART

    Returns
        Zero indicates that a RunHistRow record has been created to represent
        the subplan. -1 is returned from a RESTART call if the a RunHistRow
        already exists for the plan/subplan and is marked COMPLETE.
    """

    conn = db_connect(db_path)
    csr  = conn.cursor()
    skip = False

    # skip previously completed work if RESTART
    if run_type == RunType.RESTART:
        stmt = _select_row_from_runhist.format(plan_id, subplan_id)
        execute_sql_stmt(conn, stmt, cursor=csr)
        row = csr.fetchone()

        if row:
            plan_rec = RunhistRow._make(row)
            if plan_rec.status == RunStat.COMPLETE.name:
                skip = True

    # construct/reinit a new runhist record 
    if not skip:
        currtime = datetime.now()
        start_time = currtime.strftime("%m/%d/%Y-%H:%M:%S")

        stmt = _insupd_scheduled_runhist.format(
            plan_id,
            subplan_id,
            RunStat.SCHEDULED.name,
            start_time
        )

        execute_sql_stmt(conn, stmt, cursor=csr)

    csr.close()
    conn.commit()
    conn.close()

    if skip:
        return -1
    else:
        return 0


def stop_subplan(db_path, plan_id=None, subplan_id=None, comp_info_dict={}):
    """Complete the execution of a subplan.

    This function updates the RunHistRow record created by start_subplan()
    updating the status to COMPLETE, the completion timestamp, and "user
    fields" (such as MAE, MSE, R2) returned by the model.

    A comp_dict dictionary is populated with the names and default values
    for columns implemented in the RunHistRow table. Values matching those
    names are extracted from the comp_info_dict are written to the table. 

    TODO It might be nice to take all of the unmatched fields from comp_info_dict
    and write them into a single RunHistRow text field as key=value, ...

    Args
        db_path:        plan management database path (relative or absolute)
        plan_path:      JSON plan file (relative or absolute)
        plan_id:        the plan identifier returned by plan_prep()
        comp_info_dict: supplemental completion data dictionar
    """

    conn = db_connect(db_path)
    currtime = datetime.now()
    stop_time = currtime.strftime("%m/%d/%Y-%H:%M:%S")

    comp_dict = dict(mae=0.0, mse=0.0, r_square=0.0, weights_fn='N/A')
    _acquire_actuals(comp_dict, comp_info_dict)

    stmt = _insupd_completed_runhist.format(
    # column values
        RunStat.COMPLETE.name,
        stop_time,
        comp_dict['mae'],
        comp_dict['mse'],
        comp_dict['r_square'],
        comp_dict['weights_fn'],
    # key spec    
        plan_id,
        subplan_id
    )

    execute_sql_stmt(conn, stmt)
    conn.commit()
    conn.close()


def get_subplan_runhist(db_path, plan_id=None, subplan_id=None):
    """Return the RunHistRow record for a given plan/subplan.

    Args
        db_path:        plan management database path (relative or absolute)
        plan_id:        the plan identifier returned by plan_prep()
        subplan_id      the subplan identifier ex. '1 4.8'

    Returns
        The RunHistRow associated with the given plan/subplan is returned if
        found.
    """
    conn = db_connect(db_path)
    stmt = _select_row_from_runhist.format(plan_id, subplan_id)
    csr  = conn.cursor()
    execute_sql_stmt(conn, stmt, csr)
    row = csr.fetchone()

    if not row:
        plan_rec = None
    else:
        plan_rec = RunhistRow._make(row)

    return plan_rec

def _acquire_actuals(dft_dict, actuals_dict):
    for key, value in dft_dict.items():
        if key in actuals_dict:
            dft_dict[key] = actuals_dict[key]

def _get_planstat_key(plan_path):
    """Extract the name portion of a plan from a filepath."""
    basename = os.path.basename(plan_path)
    basepfx = basename.split(sep='.')
    return basepfx[0]


def _delete_runhistory(conn, plan_id):
    """Delete RunHistRows containing the given plan_id."""
    csr  = conn.cursor()
    stmt = _delete_from_runhistory.format(plan_id)
    execute_sql_stmt(conn, stmt, cursor=csr, trap_exception=True)
    rowcount = csr.rowcount
    print("CLEANUP processing removed %d run history records" % rowcount)
    csr.close()
    return rowcount


#------------------------------------------------------------------------------
# Plan navigation, content retrieval 
#------------------------------------------------------------------------------

def load_plan(filepath):
    """Load a JSON transfer learning plan.

    The named JSON tranfer learning plan file is loaded in a manner that preserves
    the entry order imposed when the plan was created. This allows the root entry
    to be easily located regardless of the plan entry naming scheme in use.

    Args
        filepath:   A relative or absolute path to the JSON file.

    Returns
        An entry-ordered plan in OrderedDict format is returned.
    """

    with open(filepath, 'r') as f:
        ordered_plan_dict = json.load(f, object_pairs_hook=OrderedDict)
    return ordered_plan_dict

def get_plan_create_date(plan_dict):
    _, value = _get_first_entry(plan_dict)
    return value['create_date']

def get_plan_fs_names(plan_dict):
    _, value = _get_first_entry(plan_dict)
    return value['fs_names']

def get_plan_fs_parts(plan_dict):
    _, value = _get_first_entry(plan_dict)
    return value['fs_parts']

def get_plan_nbr_subplans(plan_dict):
    _, value = _get_first_entry(plan_dict)
    return value['nbr_subplans']

def _get_first_entry(ordered_dict):
    key, value = next(iter(ordered_dict.items()))
    return key, value

def get_subplan(plan_dict, subplan_id=None):
    """Retrieve the content of a named subplan or the root plan.

    Args
        plan_dict:      The plan dictionary as returned by load_plan().
        subplan_id:     The name of the desired subplan. Omit this arg to acquire
                        the content and name of the plan tree root.

    Returns
        A (content, subplan_id) pair is returned. The returned name is useful when
        using default arguments to retrieve the root plan.
    """

    if subplan_id is None:
        subplan_id, content = _get_first_entry(plan_dict)
    else:
        content = plan_dict.get(subplan_id)
    return content, subplan_id


def get_predecessor(plan_dict, subplan_id):
    """Acquire the name of the predecessor (parent) of a given subplan.

    The plan tree is a true tree. All subplans have exactly one
    predecessor/parent. Use this function to walk 'up' the tree.

    Args
        plan_dict:      The plan dictionary as returned by load_plan().
        subplan_id:     The name of the target subplan.

    Returns
        The name of the parent subplan is returned. If the root plan name
        is specified None is returned.
    """

    segments = subplan_id.split(sep='.')
    if len(segments) <= 1:
        subplan_id = None
    else:
        segments.pop()
        subplan_id = '.'.join(segments)
    return subplan_id


def get_successors(plan_dict, subplan_id):
    """Acquire the names of the successors (children) of a given subplan.

    All subplans other than 'leaf' subplans have at least one successor. Use
    this function to walk 'down' one or more plan subtrees.

    Args
        plan_dict:      The plan dictionary as returned by load_plan().
        subplan_id:     The name of the target subplan.

    Returns
        A list of the names of all successors (children) of the given subplan
        is returned. The list may be empty.
    """
    successor_names = []
    for i in it.count(start=1):
        new_name = subplan_id + '.' + str(i)
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

    Returns
        args.plan_dict contains a dictionary representing the plan. This may be
        JSONized.

        The number of planning steps (nbr of subplans in the plan tree) is explicitly
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
        qualified_name = 'plangen_' + label + '.json'
        json_file_name = os.path.join(args.out_dir, qualified_name)
        json_abspath = os.path.abspath(json_file_name)
        write_dict_to_json(args.plan_dict, json_abspath)
        print("%s JSON file written" % json_abspath)

    if args.debug:
        pp(args.plan_dict, width=160)

    if args.test:
        test(json_abspath)

#----------------------------------------------------------------------------------
# test plan navigation and subplan entry retrieval
#----------------------------------------------------------------------------------

def test(plan_path):
    db_path = "test_sql_database"
    run_type = RunType.RESTART

    plan_id = plan_prep(db_path, "plangen_cell8-p2_drug8-p2.json", run_type)

    if (plan_id < 0):
        sys.exit("Terminating due to database detected error")

    print("\nBegin plan navigation and subplan retrieval test")
    plan_dict = load_plan(plan_path)

    # plan root name, value returned when subplan_id= is omitted
    metadata, root_name = get_subplan(plan_dict)

    # the root has no parent / predecessor
    parent_name = get_predecessor(plan_dict, root_name)
    print("Demonstrate that root \'%s\' predecessor is not defined: %s" % (root_name, parent_name))

    # the root contains metadata, it is not a run specification
    successor_names = get_successors(plan_dict, root_name)
    print("\nThe first runable configurations are defined in %s\n" % successor_names)

    # the root is the predecessor of these first level runables
    for sname in successor_names:
        parent_name = get_predecessor(plan_dict, sname)
        print("The parent of %s is %s" % (sname, parent_name))

    # run the right subtree
    print("\nRun the rightmost subtree \n")
    for i in it.count(start = 1):
        listlen = len(successor_names)
        if listlen == 0:
            break

        for name in successor_names:
            status = start_subplan(
                db_path,
                plan_path,
                plan_id=plan_id,
                subplan_id=name,
                run_type=run_type
            )

            if status < 0:
                print("subplan: %s skipped, previously processed" % name)

        select_one  = successor_names[listlen - 1]
        parent_name = get_predecessor(plan_dict, select_one)
        print("%-16s is a successor of %-16s - all successors: %s" % (select_one, parent_name, successor_names))

        value,_ = get_subplan(plan_dict, select_one)
        train_set, validation_set = parse_plan_entry(value)

        # print the coarsely parsed plan entry train/validation components
        if i == 1:
            print("\n*** Training set ***")
            pp(train_set, width=160)
            print("\n*** Validation set ***")
            pp(validation_set, width=160)
            print(" ")

        # test retrieval api
        row = get_subplan_runhist(db_path, plan_id=plan_id, subplan_id=select_one)
        print(row)

        # post subplan termination
        completion_status = dict(mse=1.1, mae=2.2, r_square=.555)

        stop_subplan(
            db_path,
            plan_id=plan_id,
            subplan_id=select_one,
            comp_info_dict=completion_status
        )

        successor_names = get_successors(plan_dict, select_one)

    print("\nEnd of branch reached")
#   plan_remove(db_path, "plangen_cell8-p2_drug8-p2.json")

#----------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

