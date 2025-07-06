# AutoCRAT
Automated Chromosome Replication Analysis & Tracking
---
[![DOI](https://zenodo.org/badge/1014308195.svg)](https://doi.org/10.5281/zenodo.15817321)


## What does it do?

AutoCRAT is a Python pipeline for analyzing live-cell imaging data that includes fluorescently labeled chromosomal loci, and extracting rich information about DNA replication and related biological processes. AutoCRAT is specifically designed for the study of replisome progression using fluorescent repressor-operator systems (as first described in [Dovrat et al., 2018](https://www.cell.com/cell-reports/fulltext/S2211-1247(18)30913-6)) and related approaches. However, it is also more generally useful for efficiently and conveniently tracking, quantifying and characterizing dots or foci in live-cell imaging data, particularly under challenging imaging conditions with low signal-to-noise ratios, and when multiple dots are simultaneously imaged in multiple fluorescent channels and the relationships between them are of interest.

## Design principles

AutoCRAT is designed to be:
- **Fully automated.** AutoCRAT does not have a GUI or API, and is intentionally non-interactive. Users define their specific data analysis needs by inputting parameters in a config file, and simply run the pipeline on a movie. AutoCRAT performs the desired analysis steps while adapting itself to the specific characteristics of your data, and the final results are returned in conveniently-structured Excel files. Once a proper config file has been prepared, AutoCRAT can analyze large amounts of data with no human intervention, taking you from raw microscopy images to publishable insights in a single click.
- **Modular and flexible**. While AutoCRAT was originally built for answering specific scientific questions using specific experimental systems, it offers flexible and adaptable capabilities for a wide variety of applications related to fluorescent dots in single or multi-channel 3D timelapse imaging data. Its core functionality involves identifying, tracking and quantifying the intensity of dots over time, as well as aligning dots in different fluorescent channels. On top of that, it includes several modules dedicated to extracting more specific insights. It is also extensible, and multiple [accessory scripts](https://github.com/dovratd/AutoCRAT-accessory-scripts) for various purposes are available. 
- **Fast**. AutoCRAT leverages multiple optimizations, as well as Python's multi-processing capabilities and [numba](https://numba.pydata.org), to process very large imaging datasets at lightning speeds.

## How do I use it?

A detailed user manual will come soon!

## Who should we blame?

AutoCRAT was written by Daniel Dovrat at the [Aharoni lab, BGU](https://lifewp.bgu.ac.il/aaharoni/). 
Among other things, it relies heavily on the excellent [PIMS](https://github.com/soft-matter/pims) and [TrackPy](https://github.com/soft-matter/trackpy). 
