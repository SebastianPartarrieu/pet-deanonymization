# De-anonymization through face recognition on PET data &middot; [![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](https://github.com/SebastianPartarrieu/live-kinect/blob/master/LICENSE)

It is fairly well known that CT and MRI scans, though de-identified, can potentially be re-identified by reconstructing the faces of the patients and using powerful facial recognition software. This could potentially be a privacy and security hazard as many public datasets have provided de-identified patient scans without adequately modifying patient facial features. Although this is changing with specific 'de-facing' software now being provided for CT and MRI scans, this issue remains largely unexplored for PET scans. At first, it seems that due to the inherent lower resolution of PET imaging there might be less of a problem as any facial reconstruction would be too noisy to exploit.

In this repository, we explore the potential methods that might be employed to try and de-anonymize pet images through morphological reconstruction, denoising and face recognition. We use an openly available dataset subject to a TCIA license agreement that was provided as part of the [AutoPET](https://autopet.grand-challenge.org/) challenge. We do not provide the data here as it is subject to the Data Usage Agreement (DUA) as detailed by the TCIA license.

In an ideal setting, we would have access to some of the patient's real-life photos to try and perform the matching ourselves and quantify re-identification performance, but we do not have a patient cohort. As a substitute, we use unsupervised metrics such as the number of patients where we can accurately locate the face using standard face detection algorithms, or those where we can accurately place facial landmarks which are a key component of various face detection software, how well we can place these landmarks and finally if we can correctly match a PET scan to its corresponding CT scan. Of course, no patient faces are shared here.

## Installation

### Getting started

Create a virtual environment using your [favorite method](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) (venv, conda, etc...) but make sure to install the dependencies using the provided environment.yml file.
If you feel like using conda: `conda env create -f environment.yml` should do the trick. Note that most of the visualizations were run remotely using Colab, so you don't necessarily need to run a local install, you can just access the notebooks directly on Colab.

### Download datasets

Follow instructions details [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287#932582870c7caa21e8b840a393398eeda1279f3b)](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258287#932582870c7caa21e8b840a393398eeda1279f3b). Note that there is a substantial amount of data (~350Gb of data) so make sure you have enough space.

### System requirements

Most of the code here was run on Colab, any working python installation >=3.8 should do the trick.

## Experiment details

The main pipeline follows: 3D voxel representation -> adaptive thresholding to keep relevant voxels -> selecting largest connected component -> generating mesh representations of isosurface using marching cubes algorithm -> ray casting to get a 2D image representation (optional -> apply image denoising technique to 2D or 3D mesh) -> applying landmark placement algorithms for potential de-anonymization -> calculate metrics.

### Reproducibility

We cannot provide to the raw data since this would breach the DUA. However, we provide the code to run our experiments here.

To reproduce the different steps, perform:

```python
python perform_morphological_reconstruction.py --petct_path="../data/raw_petct/" --save_dir="../data/"
```

```python
python perform_crossval.py --n_epochs=50 --ct_files=../data/cts/ --pet_files=../data/pets/ --save_perfs_dir=../data/training_metrics
```

If you want to reuse precomputed splits, you can specify the file path in `--splits_file`.

### Landmark placement and matching

Landmark placement to create a Face Mesh is done using Google's [Mediapipe](https://google.github.io/mediapipe/). We need to be careful as these pipelines are tuned for RGB images and we are working with grayscale.

To perform landmark placement, run :

```python
python perform_landmark_placement.py --ct_files="../data/cts/" --pet_files="../data/pets/" --splits_file="../model_checkpoints/fold_splits.npy" --models_path="../model_checkpoints/" --save_dir="../data/landmarks/"
```

To perform landmark matching, run :

```python
python perform_landmark_matching.py --landmarks_files="../data/landmarks/" --n_folds=5 --save_dir="../data/nearest_neigh_landmarks_repositioning/" --icp
```

If you omit the `--icp` parameter, the pipeline will run without the alignement step.
