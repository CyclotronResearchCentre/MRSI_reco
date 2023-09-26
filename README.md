# SCAIFIELD MRSI analysis

## Prerequisite

- This tool performs the SCAIFIELD MRSI analysis
- Some third party software
    - LCModel
    - FSL

## Usage

just call 

```
python mrsi_analysis.py [-h] [--path PATH] [--site SITE] [--sub SUB] [--ses SES]
  --path PATH  path #BIDS folder (default: None)
  --site SITE  site (default: None)
  --sub SUB    subject name (default: None)
  --ses SES    session name (default: None)
```

## Output

You can find the output in

```path/derivatives/site/sub/ses/mrsi/lcm```
```path/derivatives/site/sub/ses/mrsi/maps```

- lcm
    - LCModel input and output of each voxel within brain mask
- maps
    - Metabolites' maps (in .nii format)
    - FWHM/SNR maps

## Working principle

1. Read in all .dcm / .IMA files
    - Sort for z-Position (code borrowed from suspect)
2. Appy Hamming filter
3. Create dummy nii file (in MRSI image space)
3. Use MPRAGE to create brain mask
    - register to MRSI image space (only quaternions)
5. Perform LCModel quantification for all voxels in brain mask
6. Read in LCModel files and create maps

## TODO:

- Pseudo single-voxel
