# SCAR_test #

This is the repository for the paper "Verifying Selected Completely At Random (SCAR) assumption in Positive-Unlabeled Learning", accepted on the main track at the ECAI 2024 conference.

The repository contains source code,  datasets used for experiments and the supplement.

## Abstract ##

The goal of positive-unlabeled (PU) learning is to train a binary classifier on the basis of training data containing positive and unlabeled instances, where unlabeled observations can belong either to the positive class or to the negative class. Modeling PU data requires certain assumptions on the labeling mechanism that describes which positive observations are assigned a label. The simplest assumption, considered in early works, is SCAR (Selected Completely at Random Assumption), according to which the propensity score function, defined as the probability of assigning a label to a positive observation, is constant. Alternatively, a much more realistic assumption is SAR (Selected at Random), which states that the propensity function solely depends on the observed feature vector. SCAR-based algorithms are much simpler and computationally much faster  compared to SAR-based algorithms, which usually require challenging estimation of the propensity score. In this work, we propose a relatively simple and computationally fast test that can be used to determine whether the observed data meet the SCAR assumption. Our test is based on generating artificial labels conforming to the SCAR scenario, which in turn allows  to mimic the distribution of the test statistic under the null hypothesis of SCAR.  We justify our method theoretically.
In experiments, we demonstrate that the test successfully detects various deviations from SCAR scenario and at the same time it is possible to effectively control the type I error. The proposed test can be recommended as a pre-processing step to decide which final PU algorithm to choose in cases when nature of labeling mechanism is not known.

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




