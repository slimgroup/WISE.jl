<h1 align="center">WISE: full-Waveform variational Inference via Subsurface Extensions</h1>

Code to reproduce results in Ziyi Yin\*, Rafael Orozco\*, Mathias Louboutin, Felix J. Herrmann, "[WISE: full-Waveform variational Inference via Subsurface Extensions](https://slim.gatech.edu/content/wise-full-waveform-variational-inference-subsurface-extensions)".

## Software descriptions

All of the software packages used in this paper are fully *open source, scalable, interoperable, and differentiable*. The readers are welcome to learn about our software design principles from [this open-access article](https://library.seg.org/doi/10.1190/tle42070474.1).

#### Wave modeling

We use [JUDI.jl](https://github.com/slimgroup/JUDI.jl) for wave modeling and inversion, which calls the highly optimized propagators of [Devito](https://www.devitoproject.org/).

#### Conditional normalizing flows

We use [InvertibleNetworks.jl] to train the conditional normalizing flows (CNFs). This package implements memory-efficient invertible networks via hand-written derivatives. This ensures that these invertible networks are scalable to realistic 3D problems.

## Installation

First, install [Julia](https://julialang.org/) and [Python](https://www.python.org/). The scripts will contain package installation commands at the beginning so the packages used in the experiments will be automatically installed.

## Scripts

There are 4 main scripts with descriptions below.

[gen_cig_openfwi.jl](scripts/gen_cig_openfwi.jl) generates seismic data and computes common-image gathers for the CurveFault-A velocity models in the [Open FWI dataset](https://arxiv.org/abs/2111.02926). [train_openfwi.jl](scripts/train_openfwi.jl) trains the conditional normalizing flows with pairs of velocity models and (extended) reverse-time migrations for the Open FWI dataset.

[gen_cig_compass.jl](scripts/gen_cig_compass.jl) generates seismic data and computes common-image gathers for the velocity models in the [Compass dataset](https://doi.org/10.3997/2214-4609.20148575). [train_compass.jl](scripts/train_compass.jl) trains the conditional normalizing flows with pairs of velocity models and (extended) reverse-time migrations for the Compass dataset.

The script [utils.jl](scripts/utils.jl) parses the input as keywords for each experiment.

## LICENSE

The software used in this repository can be modified and redistributed according to [MIT license](LICENSE).

## Reference

If you use our software for your research, we appreciate it if you cite us following the bibtex in [CITATION.bib](CITATION.bib).

## Authors

This repository is written by [Ziyi Yin] and [Rafael Orozco] from the [Seismic Laboratory for Imaging and Modeling] (SLIM) at the Georgia Institute of Technology.

If you have any question, we welcome your contributions to our software by opening issue or pull request.

SLIM Group @ Georgia Institute of Technology, [https://slim.gatech.edu](https://slim.gatech.edu/).      
SLIM public GitHub account, [https://github.com/slimgroup](https://github.com/slimgroup).    

[license-status]:LICENSE
[license-img]:http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat?style=plastic
[Seismic Laboratory for Imaging and Modeling]:https://slim.gatech.edu/
[InvertibleNetworks.jl]:https://github.com/slimgroup/InvertibleNetworks.jl
[Ziyi Yin]:https://ziyiyin97.github.io/
[Rafael Orozco]:https://slim.gatech.edu/people/rafael-orozco