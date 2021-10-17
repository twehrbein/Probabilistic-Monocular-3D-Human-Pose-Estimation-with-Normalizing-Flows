# Probabilistic-Monocular-3D-Human-Pose-Estimation-with-Normalizing-Flows
This is the official implementation of the ICCV 2021 Paper "Probabilistic Monocular 3D Human Pose Estimation with Normalizing Flows" by Tom Wehrbein, Marco Rudolph, Bodo Rosenhahn and Bastian Wandt.

## Installation instructions
We recommend creating a clean [conda](https://docs.conda.io/) environment. You can do this as follows:
```
conda env create -f environment.yml
```

After the installation is complete, you can activate the conda environment by running:
```
conda activate ProHPE
```

Finally, [FrEIA](https://github.com/VLL-HD/FrEIA) needs to be installed:
```
pip install git+https://github.com/VLL-HD/FrEIA.git@ad2f5a261a2fc0002fb4c9adeff7a62b0e9dd4e1
```

## Data
Download the used Human3.6M detections and Gaussian fits from [Google Drive](https://drive.google.com/file/d/1eE02AfGSPW2vUY7OlCaolrM9ts87ONNM/view?usp=sharing). Afterwards, extract the zip file to the data/ directory.

Due to licensing, it is not possible for us to provide any data from Human3.6M. Therefore, the 3D pose data needs to be downloaded from the official dataset [website](http://vision.imar.ro/human3.6m/description.php) (account required). For each subject, download the file 'D3_Positions_mono' and extract it to the data/ directory. Afterwards, run
```
python create_data.py
```
to automatically merge the detections and Gaussian fits with the 3D ground truth data.

For information about the dataset structure, see data/data_info.txt.

## Run evaluation code
Evaluate the pretrained model on the whole testset of Human3.6M:
```
python eval_action_wise.py --exp original_model
```
Evaluate on the hard subset of Human3.6M containing highly ambiguous examples:
```
python eval_whole_set.py --exp original_model
```

## Run training code
Training can be started with:
```
python train.py --exp experiment_name
```

## Citation
Please cite the paper in your publications if it helps your research:
    
    @inproceedings {WehRud2021,
      author = {Tom Wehrbein and Marco Rudolph and Bodo Rosenhahn and Bastian Wandt},
      title = {Probabilistic Monocular 3D Human Pose Estimation with Normalizing Flows},
      booktitle = {International Conference on Computer Vision (ICCV)},
      year = {2021},
      month = oct
    }

Link to the paper:

- [Probabilistic Monocular 3D Human Pose Estimation with Normalizing Flows](https://arxiv.org/abs/2107.13788)
