# Kinetics Model- Km, Kcat Estimation Workflow

We created a video showcasing how to use the code — please visit our [wiki page](https://2025.igem.wiki/technion-israel/model) for a step-by-step walkthrough video.

This repository contains scripts and resources to guide you through the process of estimating kinetic parameters (Km, Kcat) of your protein of interest. Below is an overview of the workflow.

---
The model can be fit to test data for multiple substrates — please feel free to input data for as many substrates as needed.  

## step 1 – Proof Check of the Model - generating input for kinetic model

Before running kinetic estimations, you should verify that the model can reproduce results for your protein of interest.  
For this purpose, use **`values_of_Pt_generated_research.py`** file first.
 
  This script takes kinetic parameters reported in the literature and uses them to generate simulated curves for your protein.  

0. Extract known values of **Km**, **kcat**, and **Vmax** from literature sources related to your substrate or substrates of choice.
1. Insert these values into the script, following the input template provided in the code.  
2. Set up your assay conditions (e.g., substrate concentration range, time resolution).
   - We recommend starting with conditions similar to those described in the literature.  
   - For `dt_sec` (time step), test several values to identify the most stable fit for your project. 
3. Choose the output folder where results will be stored.
4. Define subfolder names for the generated CSV and PNG outputs.
5. Ensure your units match the conventions expected by the model.

**The final results of script will include:** simulated curves for your input substrates, as well as data of **dp/dt visualization** for each run, all in automatically organized folders containing both CSV data and PNG plots.

---

## step 2 - Proof Check of the Model - running kinetics model on generated values

After generating test data, you can run **`kinetics_model.py`** with the CSV produced in Step 1.  
The script requires a **JSON file** as input.
 
### Expected JSON structure  
The JSON must include experiment details and Approximate Bayesian Computation (ABC) settings:
 - **Experiment details**  
    - `E_0`: Initial enzyme concentration (mM).  
    - `S_0`: Initial substrate concentration (µM).  
    - `h`: Time step size (Δt) for numerical integration (min).  
    - `exp_last`: Total duration of the experiment (min).  

 - **Parameter bounds**  
    - `low_K_M` / `high_K_M`: Search range for Km.  
    - `low_k_cat` / `high_k_cat`: Search range for kcat.  
    - *Tip:* Test several thresholds to identify the most stable fit.  

- **ABC settings**  
    - `N`: Number of candidate parameter samples per iteration.  
    - `No_sim`: Number of stochastic simulations per candidate.  
    - `points_prior`: Number of points used to construct the prior distribution.  
    - *Tip:* Use the default example values, which were optimized by Tomczak & Węglarz-Tomczak (2019).  

### Example JSON
```json
{
  "file_name": {
    "file": "/path/to/your_csv_file_generated.csv"
  },
  "experiment_details": {
    "E_0": 4.7001e-3, 
    "S_0": 100.0,
    "h": 0.25,
    "exp_last": 2.0
  },
  "abc_details": {
    "low_K_M": 600.0,
    "high_K_M": 1000.0,
    "low_k_cat": 31.5167,
    "high_k_cat": 52.5167,
    "N": 2000,
    "No_sim": 400,
    "points_prior": 300
  }
}
```

**This step is important** because if the model correctly reproduces expected behavior, it increases confidence that it will work properly for new experimental data.

---

## step 3 - Designing a Wet-Lab Protocol

Once the model is validated, you need to generate experimental data under conditions that match the assumptions of the model.

  1. Prepare a wet-lab protocol for substrate concentration gradients and activity measurements.  
  2. if necessary convert OD to P(t) using Beer-lambert's law.
  - Ensure that the experimental readout (e.g., absorbance, activity units) corresponds to the variables expected in the model.  

 We provide an example protocol in our [wiki protocols section](https://2025.igem.wiki/technion-israel/protocols), which can be adapted to your own lab conditions.  
**Tip**: Consistency between the experimental setup and the model inputs is critical for reliable parameter estimation.

---

## step 4 - Preparing the Inputs for the kinetic model and results

1. Create a CSV file from your lab data P(t). Make sure it matches the JSON conditions and lab setup (e.g., number of readouts and their time step).  
2. Update your JSON file: ensure it calls the correct CSV file you generated from lab results.  
3. Run the model:  

```bash
python kinetics_model.py input.json
```
**The outputs will include** estimated **Km** and **kcat** values with mean and standard deviation, results printed in the console as well as results saved into a `results.txt` file.  
  
---