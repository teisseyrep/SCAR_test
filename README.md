# SCAR #

This is the repository for the paper "Verifying Selected Completely At Random (SCAR) assumption in Positive-Unlabeled Learning".

The repository contains source code,  datasets used for experiments and the supplement.

## Supplement ##

The file ``supplement.pdf`` contains additional information for the work: proofs of some theorems, experimental results and information about data sets.

## Data ##

Tabular datasets can be found in ``/data`` folder. File ``prepare_image_data.py`` contains a script for preprocessing image datasets. After running ``prepare_image_data.py`` script, data files will be saved in  ``/data`` folder in csv format.

## Experiments ##

The file ``make_exp_art_data.py`` contains code to run experiments for artificial datasets. The experiment allows to estimate the probability of rejecting the null hypothesis for the 4 considered statistics (KL, KLCOV, KS, NB AUC) and various values of the n and g parameters. The results are saved in the ``/results`` folder.

The file ``make_exp_real_data.py`` contains code to run experiments for real datasets. The experiment allows to estimate the probability of rejecting the null hypothesis for the 4 considered statistics (KL, KLCOV, KS, NB AUC) and various values of parameter g. The results are saved in the ``/results`` folder.

## Example usage ##

We suggest to create a new environment and activate it using:

```bash
 $ conda env create -f environment.yml
 $ conda activate scar
  ```

An example usage of the test on PU data can be found in the notebook ``example_usage.ipynb``




