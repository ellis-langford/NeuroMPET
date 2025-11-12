<div align="center">
  <img src="./assets/neuro_mpet_logo.png" width="700">
  <br><br>
  <p align="center"><strong>Neuro Multiple-Network Poroelastic Theory: Meshing & Modelling</strong></p>
</div>

<div align="center" style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-top: 10px;">
  <a href="https://profiles.ucl.ac.uk/101480-ellis-langford"><img src="https://custom-icon-badges.demolab.com/badge/UCL Profile-purple?logo=ucl" alt="UCL Profile"></a>
  <a href="https://orcid.org/0009-0006-1269-2632"><img src="https://img.shields.io/badge/ORCiD-green?logo=orcid&logoColor=white" alt="ORCiD"></a>
  <a href="https://github.com/ellis-langford"><img src="https://img.shields.io/badge/GitHub-%23121011.svg?logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://uk.linkedin.com/in/ellis-langford-8333441ab"><img src="https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff" alt="LinkedIn"></a>
</div>

## Introduction

NeuroMPET is a processing workflow that: preprocesses, registers, segments a NIfTI image; then generates meshes, maps tissue classes onto the mesh and models using the MPET solver.

The core solver code was developed by Liwei Guo (liwei.guo@ucl.ac.uk) and Yiannis Ventikos (y.ventikos@ucl.ac.uk) at University College London. To request access to the MPET solver core code please contact Ellis Langford (ellis.langford.19@ucl.ac.uk).


## Requirements

To successfully run the NeuroMPET pipeline, please ensure the following requirements are met:

**Ubuntu 22.04 + Docker 27.3.1 + Python 3.10**<br>
*(other versions may be compatible but have not been tested)*


## Installation & Quick Start

To install the necessary components for NeuroMPET, please follow the steps below:

- Either, pull the docker image from GitHub container registry:

  ```bash
  docker pull ghcr.io/ellis-langford/neuro_mpet:v1
  ```

- Or clone the code from the GitHub repo and build image yourself:
  
  ```bash
  git clone https://github.com/ellis-langford/NeuroMPET.git
  cd NeuroMPET
  docker build -t ghcr.io/ellis-langford/neuro_mpet:v1 .
  ```
  
- Lauch a docker container from the NeuroMPET docker image:
  
  ```bash
  docker run -it -v /path/to/data:/path/to/data ghcr.io/ellis-langford/neuro_mpet:v1 bash
  ```

- Edit the example properties file to suit your requirements
  
  ```bash
  nano example_properties_file.json
  ```

- Navigate to your chosen output directory:
  
  ```bash
  cd /output_dir
  ```

- Run the pipeline:
  
  ```bash
  python3.10 /app/src/main.py --input_im /path/to/input/dir --props_fpath /path/to/properties/file

## Citation
The following papers should be cited when this code is used:
```
1. Tully, B. and Ventikos, Y. (2011).<br>
   Cerebral water transport using multiple-network poroelastic theory: application to normal pressure hydrocephalus.<br>
   Journal of Fluid Mechanics, 667:188-215.<br>
2. Guo, L., Vardakis, J. C., Lassila, T., Mitolo, M., Ravikumar, N., Chou, D., … Ventikos, Y. (2018).<br>
   Subject-specific multi-poroelastic model for exploring the risk factors associated with the early stages of Alzheimer’s disease.<br>
   Interface Focus, 8:20170019.<br>
3. Guo, L., Vardakis, J. C., Chou, D., and Ventikos, Y. (2020).<br>
   A multiple-network poroelastic model for biological systems and application to subject-specific modelling of cerebral fluid transport.<br>
   International Journal of Engineering Science, 147:103204.<br>
```