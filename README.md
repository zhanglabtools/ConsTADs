## 1. ConsTADs

Topologically associating domains (TADs) have emerged as basic structural and functional units of genome organization, and have been determined by many computational methods from Hi-C contact maps. However, the TADs obtained by different methods vary greatly, which makes the accurate determination of TADs a challenging issue and hinders subsequent biological analyses about their organization and functions. This project is about comparing different TAD-calling methods, building the TAD separation landscape and finding the consensus TADs from results of multiple methods.

## 2. Workflow

This computational framework consists of three main steps, including:

1. Running different TAD-calling methods on the same Hi-C interaction profile;
2. Collecting the TAD boundaries identified by each method and performing boundary voting;
3. Refining the boundary score profile based on the contrast P-values of chromatin interactions using three operations, Add, Filter and Combine, to construct the TAD separation landscape.

<img src="https://github.com/dangdachang/ConsTADs/raw/main/TAD%20separation%20landscape%20construction.png" style="zoom:60%;" />

The TAD separation landscape can be used in scenarios such as:

1. Comparing domain boundaries across multiple cell types for discovering conserved and divergent topological structures;
2. Deciphering three types of boundary regions with diverse biological features;
3. Identify <u>Cons</u>ensus <u>T</u>opological <u>A</u>ssociating <u>D</u>omain<u>s</u> (ConsTADs).

<img src="https://github.com/dangdachang/ConsTADs/raw/main/TAD%20separation%20landscape%20application.png" style="zoom:60%;" />

