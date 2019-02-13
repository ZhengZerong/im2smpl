# im2smpl
Predicts a SMPL model given a RGB image using clues from HMR, Simplify, AlphaPose, etc. 

### Setup
1. clone this repository:
```sh
git clone https://github.com/ZhengZerong/im2smpl.git
cd ./im2smpl
```

2. Setup AlphaPose in `./AlphaPose/` according to [this guidance](https://github.com/MVIG-SJTU/AlphaPose/blob/master/README.md#installation). 
After that, run the official demo script (`./AlphaPose/run.sh`) to make sure AlphaPose is properly setup. 

3. Setup HMR in `./HMR/` according to [this guidance](https://github.com/akanazawa/hmr/blob/master/README.md). 
After that, run the official demo script (`./HMR/demo.py`) to make sure HMR is properly setup. 

4. Setup LIP in `./LIP_JPPNet/` according to [this guidance](https://github.com/Engineering-Course/LIP_JPPNet/blob/master/README.md). 
After that, run the official demo script (`./LIP_JPPNet/evaluate_parsing_JPPNet-s2.py`) to make sure LIP is properly setup.  

### Usage
Run the following command:
```sh
python main.py --img_file ./path/to/image --out_dir ./path/to/output/directory/
# for example: python main.py --img_file ./example/1.png --out_dir ./example/
```
Or modify the parameters in the provided script `main.sh` and simply run:
```sh
sh main.sh
```

### Citation
If you find the code useful in your work, you should cite the following papers:
```
@inproceedings{Bogo:ECCV:2016,
  title = {Keep it {SMPL}: Automatic Estimation of {3D} Human Pose and Shape
  from a Single Image},
  author = {Bogo, Federica and Kanazawa, Angjoo and Lassner, Christoph and
  Gehler, Peter and Romero, Javier and Black, Michael J.},
  booktitle = {Computer Vision -- ECCV 2016},
  series = {Lecture Notes in Computer Science},
  publisher = {Springer International Publishing},
  month = oct,
  year = {2016}
}      

@inProceedings{kanazawaHMR18,
  title={End-to-end Recovery of Human Shape and Pose},
  author = {Angjoo Kanazawa
  and Michael J. Black
  and David W. Jacobs
  and Jitendra Malik},
  booktitle={Computer Vision and Pattern Regognition (CVPR)},
  year={2018}
}

@article{liang2018look,
  title={Look into Person: Joint Body Parsing \& Pose Estimation Network and a New Benchmark},
  author={Liang, Xiaodan and Gong, Ke and Shen, Xiaohui and Lin, Liang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2018},
  publisher={IEEE}
}

@inproceedings{xiu2018poseflow,
  title = {{Pose Flow}: Efficient Online Pose Tracking},
  author = {Xiu, Yuliang and Li, Jiefeng and Wang, Haoyu and Fang, Yinghong and Lu, Cewu},
  booktitle={BMVC},
  year = {2018}
}
```
