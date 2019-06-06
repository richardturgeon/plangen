# Uno: Milestone 13 Transfer Learning 
This README discusses the use of the `plangen.py` script to partition feature sets for experiments with large scale transfer learning and parallel model training. The utility does the following:

* Accept text files containing lists of feature names of arbitray length, each is called a feature-set
* Generate unique combinations of features from each feature set, setting the stage for transfer learning (partitioning)
* Construct a tree depicting how successive, parallel training sessions can be scheduled upon the completion of a predecessor/parent (planning) 

## Overview
A number of partitioning schemes and data representation strategies have been discussed. The focus here is the configuration agreed upon at the May 2019 CANDLE hack-a-thon. Specifically:

* There are two feature sets, cell-lines and drugs.
* In a prototype implementation, each feature-set will contain eight entries. The target configuration will have 1000 cell features and 1000 drug features.
* Partitioning is accomplished by recursively splitting the cell vs drug graph into quadrants. 
* Each such partitioning presents four training opportunities, each uniquely combines three quadrants and omits one. 
* The omitted quadrant defines validation data for the training run. Partitioning/planning recurs on this quadrant to define successors.
* The four training operations can be scheduled to run in parallel once the training of their common parent completes.
* The partitioning scheme as well as the training parent/child relationships will be expressed in a JSON document.

## Running the script

`plangen.py` arguments are defined in `planargs.py`. `sample-command-line` is a script that demonstrates the parameters used to accomplish the objectives outlined above. Refer to that sample when reading the argument descriptions below. `--help` gives a brief summary of all arguments. 

The critical parameters are `--fs_names`, `--fs_paths` and `--fs_parts`. In each `fs` stands for feature_set. Each parameter is required and each must specify the same number values. `--fs_names` takes two or more values providing feature set names such as `cells` and `drugs`.

`fs_paths` takes path sepecifications for the corresponding feature-set files. All of the usual file search rules apply, they can be relative or absolute paths. Optionally, `--in_dir` can be used to provide common high-level qualification.  

`fs_parts` defines the partitioning scheme for each of the feature-sets. So in our scenario above, `--fsparts 2 2` specifies that at each iteration, both the `cells` and `drugs` feature-sets will be halved, giving the quadrants discussed above at each iteration. Non-symetric partitioning may prove useful when the number of feature-set line items diverges from the "square" model. 

`--in_dir` is optional. It can be used to simplify the coding of `--fs_paths` path names. The rules of os.path.join() apply.

`--out_dir` is optional. It can be used to place output files, the JSON format plan in particular, to a specific directory.


`--debug` is optional. If specified the final plan dictionary is pretty-printed. This is quite a bit easier to read than the JSON file.    

## Contact

Richard Turgeon
<rturgeon@anl.gov>   
Created: 2019-06-07
