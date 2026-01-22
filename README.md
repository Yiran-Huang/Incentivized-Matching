# STATISTICAL MATCHING MARKETS UNDER UNCERTAIN PREFERENCES

We provide the code for the paper. A Jupyter Notebook (in the `Paper_Outcomes` folder) is also included to directly generate the figures from the uploaded outputs, without rerunning the code.

Each experiment module is stored in a separate folder. The files “DA_algorithm.py”, “get_bound.py”, “Matching_Method.py”, and “Matching_Method_non_covariates.py” are shared across all modules and are therefore placed in the root directory. To run the code, enter the relevant folder and follow the instructions in its README file.

This `requirements.txt` file specifies the required Python dependencies. Install them using `pip install -r requirements.txt`.

It is recommended to use **Python 3.11 or later**. Based on our experience, the linear programming in `scipy` may hang under Python 3.6, 3.8, and 3.9, while this issue has not been observed under Python 3.11.13.



## Folder Structure

`vsUCB`: Comparing the performance of the Upper Confidence Bound (UCB) algorithm and the proposed Incentivized-UCB (I-UCB) algorithm. The results correspond to **Subsection 5.1** and **Subsection C.1**.

`Simulation`: Ablation of components of I-UCB. The results correspond to **Subsection 5.2**.

`Journal`: Simulation on journal recommendation. The results correspond to **Subsection 6.1**.

`Yelp`: Simulation on Yelp restaurant recommendation. The results correspond to **Subsection 6.2**.

`Cold_Start`: Simulation on Cold-start Problem. The results correspond to **Theorem 4.5**.

`processed_data`: Processed data used in the journal and Yelp restaurant recommendation simulations.

`Paper_Outcomes`: Outcomes used to produce the figures reported in the paper. **(under construction)**

`Plotting`: This folder contains scripts for generating figures from saved experimental results after the experiments have been run.  **(under construction)**



## Core Files

The following describes the core functions in the files.

`DA_algorithm.py`: Implementation of the Deferred Acceptance (DA) algorithm.

- `DA_algorithm `: Generates a matching given preferences and quotas.

`get_bound.py`: Predicts upper/lower bounds under the proposed Safe Anytime-Valid Inference (SAVI) framework.

- `update_matrix`: Constructs new constraint matrix, center vector, and band vector from newly observed data.
- `get_predict_bound`: Infers upper/lower bounds from the constraint matrix, center vector, and band vector.

`Matching_Method.py`: Matching procedure.

- `IUCB`: Implementation of the proprosed Incentivized-UCB (I-UCB) algorithm.
- `matching_procedure`: Simulated matching process.

`Matching_Method_non_covariates.py`: Matching procedure without covariates. A closed-form solution to I-UCB exists in this case, so this implementation runs much faster than "Matching_Method.py".

- `IUCB_noncovariate_procedure`: Simulated matching process using the I-UCB algorithm.
- `UCB_noncovariate_procedure`: Simulated matching process using the UCB algorithm.



## Notes

Some scripts use parallel computing. Certain experiments may take a long time to run. Please refer to the README file in each folder for details.



