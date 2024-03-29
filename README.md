**Quantifying the impact of changes in effective population size and expression level on the rate of coding sequence evolution**\
Thibault Latrille, Nicolas Lartillot,\
_Theoretical population biology_, Volume 142, pages 57-66\
https://doi.org/10.1016/j.tpb.2021.09.005

**LaTeX for the manuscript and figures are available at https://github.com/ThibaultLatrille/PhD/tree/master/GenotypePhenotypeFitness**

---

This repository is meant to provide the necessary scripts and data to reproduce the figures shown in the manuscript.
The experiments can either run on a local computer or in a cluster configuration (slurm).

The experiments are meant to run on Linux/Unix/MacOS operating systems.

If problems and/or questions are encountered, feel free to [open issues](https://github.com/ThibaultLatrille/GenotypePhenotypeFitness/issues).

## 0. Local copy
Clone the repository and cd to the dir.
```
git clone https://github.com/ThibaultLatrille/GenotypePhenotypeFitness
cd GenotypePhenotypeFitness
```

## 1. Installation

### Installation on debian
Install the compiling toolchains:
```
sudo apt install -qq -y make cmake clang
```
Clone and compile the C++ code for *SimuEvol*
```
git clone https://github.com/ThibaultLatrille/SimuEvol && cd SimuEvol && git checkout v1.0 && make release && cd ..
```
Install python3 packages
```
sudo apt install -qq -y python3-dev python3-pip screen
pip3 install snakemake numpy matplotlib statsmodels pandas ete3 --user
```

## 2. Run the simulation and reproduce the figures
To reproduce a figures of the manuscript, one can use the config files of the simulations (.yaml).
```
python3 ./scripts/simulated_experiment.py --folder Heatmap --config figure-3D.yaml --nbr_cpu 4
```
The script _simulated_experiment.py_ also contains options to run the simulations on a cluster (slurm).

To reproduce all the figures of the manuscript, this loop run all of them.
```
for FOLDER in "Heatmap" "Scaling" "Relaxation"; do
    for CONFIG in ${FOLDER}/*.yaml; do
      python3 ./scripts/simulated_experiment.py --folder${FOLDER} --config $(basename "${CONFIG}") --nbr_cpu 4
    done
done
```
Once the run are completed, the script _copy_artworks_after_run_all.sh_ copies the necessary figures in the manuscript folder.
```
sh ./copy_artworks_after_run_all.sh
```
Then the .tex files (main and supp. mat.) in the manuscript folder can be compiled with all figures.

## 3. Add features or debug in the python scripts
You made modifications to one of the python script, a notebook, this README.md, or you added new features.
You wish this work benefits to all (futur) users of this repository?
Please, feel free to open a [pull-request](https://github.com/ThibaultLatrille/GenotypePhenotypeFitness/pulls)

## 4. Add features or debug in *SimuEvol*
You made modifications to the C++ code of the simulation framework *SimuEvol*.
You wish this changes benefit to all users of these software?

Please, feel free to open pull-requests in the respective GitHub repository:
* https://github.com/ThibaultLatrille/SimuEvol 

## Licence

The MIT License (MIT)

Copyright (c) 2019 Thibault Latrille

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
