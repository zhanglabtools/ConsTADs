

# ConsTADs

Topologically associating domains (TADs) have emerged as basic structural and functional units of genome organization, and have been determined by many computational methods from Hi-C contact maps. However, the TADs obtained by different methods vary greatly, which makes the accurate determination of TADs a challenging issue and hinders subsequent biological analyses about their organization and functions. This project is about comparing different TAD-calling methods, building the TAD separation landscape and finding the consensus TADs from results of multiple methods.

## Overview

This computational framework consists of three main steps, including:

1. Running different TAD-calling methods on the same Hi-C contact map;
2. Collecting the TAD boundaries identified by each method and performing boundary voting;
3. Refining the boundary score profile based on the contrast P-values of chromatin interactions using three operations, Add, Filter and Combine, to construct the TAD separation landscape.

<p align="center">
<img src="./TAD%20separation%20landscape%20application.png" width="65%" height="65%" />

</p>

The TAD separation landscape can be used in scenarios such as:

1. Comparing domain boundaries across multiple cell types for discovering conserved and divergent topological structures;
2. Deciphering three types of boundary regions with diverse biological features;
3. Identify <u>Cons</u>ensus <u>T</u>opological <u>A</u>ssociating <u>D</u>omain<u>s</u> (ConsTADs).

<p align="center">
<img src="./TAD%20separation%20landscape%20construction.png"  width="65%" height="65%"/>


</p>

## Getting start

### Installation

It's recommended to create a conda environment:

```shell
conda create -n ConsTADs python=3.7
conda activate ConsTADs
```

Download packages

```shell
git clone https://github.com/zhanglabtools/ConsTADs.git
cd ConsTADs
```

Install required packages:

```shell
pip install -r requirement.txt
```

Install ConsTADs by PyPI:

```shell
pip install ConsTADs
```

Install from source code:

```shell
git clone https://github.com/zhanglabtools/ConsTADs.git
cd ConsTADs
python setup.py build
python setup.py install
```

### Example usage

See [ConsTADs usage.ipynb](./ConsTADs%20usage.ipynb).

### Support

If you are having issues, please let us know. We have a mailing list located at:

* dangdachang@163.com

### Citation

If ConsTADs is useful for your research, consider citing our preprint:

> Defining the separation landscape of topological domains for decoding consensus domain organization of 3D genome.
> Dachang Dang, Shao-Wu Zhang, Ran Duan, Shihua Zhang.
> bioRxiv 2022.08.08.503155; **doi:** https://doi.org/10.1101/2022.08.08.503155
