# GlaUnTI: GLAcier-UNiversal Temperature Index model

[Konstantin A. Maslov](https://people.utwente.nl/k.a.maslov), [Thomas Schellenberger](https://www.mn.uio.no/geo/english/people/aca/geohyd/thosche/), [Claudio Persello](https://people.utwente.nl/c.persello), [Alfred Stein](https://people.utwente.nl/a.stein)

[[`Paper`]()] [[`Dataset`](#dataset)] [[`BibTeX`](#citing)] 

<br/>

![results](assets/results.png)

Glacier surface mass balance (SMB) is a key climate indicator and a central driver of glacier change. 
Direct SMB observations remain sparse and unevenly distributed.
Hence, transferable SMB models are essential for large-scale assessments and projections. 
Here, we propose the GLAcier-UNiversal Temperature Index model (GlaUnTI) for this purpose. 
This hybrid physics-machine learning model modifies a fully differentiable temperature index (TI) SMB model by introducing a shallow convolutional neural corrector. 
It learns spatially and temporally varying adjustments to a small set of physically interpretable TI parameters, using glacier geometry and aggregated climate information. 
We calibrate four models&mdash;a basic TI model, a purely data-driven recurrent neural network with no physical inductive bias and two GlaUnTI variants, with and without glacier facies maps as predictors&mdash;using a dataset of 65 European glaciers spanning 1995&ndash;2024 and covering the Alps, Scandinavia, Iceland, Svalbard and the Pyrenees. 
Their performance is evaluated on a spatially independent test subset of 13 glaciers across heterogeneous regions. 
The evaluation uses 793/756/314 (annual/winter/summer) point SMB measurements and 312/235/233 glacier-wide SMB estimates. 
On the test glaciers, the baseline TI model achieves annual point-level performance with *r* = 0.854 and an RMSE equal to 1.707 m w.e. 
With GlaUnTI, *r* increases to 0.940 and the RMSE reduces to 1.068 m w.e. 
At the glacier-wide scale, the baseline TI model attains an *r* equal to 0.606 and an RMSE of 0.805 m w.e. 
With GlaUnTI, *r* increases to 0.700 and the RMSE reduces to 0.627 m w.e. 
Including glacier facies maps from the end of the ablation season to the corrector yields moderate benefits in glacier-wide summer (11.0%) and annual (12.2%) SMB estimates. 
We found that the purely data-driven baseline model overall shows the weakest spatial transferability. 
Also, end-to-end differentiability enables efficient gradient-based calibration, transfer learning, inverse optimisation of effective forcing perturbations, formal model explainability and propagation of forcing-driven aleatoric uncertainty through long SMB trajectories. 
These results demonstrate that parameter-corrected hybrid models improve SMB transferability across diverse climate regimes while preserving a physically grounded structure, suitable for integration into broader glacier evolution workflows and for informing climate-related policies. 

<br/>

## Dataset

The dataset is available at [https://doi.org/10.4121/5ea53bc3-2c85-42bb-89d1-606c8ed1d80a](https://doi.org/10.4121/5ea53bc3-2c85-42bb-89d1-606c8ed1d80a). 
Download, unzip and place it in a separate folder. 
Adjust `data_folder = ...` in `constants.py` accordingly to proceed. 


## Getting started


## License

This software is licensed under the [GNU General Public License v2](LICENSE).


## Citing

To cite the paper/repository, please use the following bib entry. 

```
@article{
  ...
}
```
