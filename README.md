# im2smpl
Predicts a SMPL model given a RGB image using clues from HMR, Simplify, AlphaPose, etc. 

### Setup
1. clone this repository:
```sh
git clone https://github.com/ZhengZerong/im2smpl.git
cd ./im2smpl
```

2. Set up my forked **AlphaPose** according to [this guidance](https://github.com/ZhengZerong/AlphaPose). 
After that, run the official demo script (`./AlphaPose/run.sh`) to make sure AlphaPose is properly setup. Please don't use the authors' repository because they have renewed the code and their implementation has compatibility issues with our code.  

3. Setup **HMR** in `./HMR/` according to [this guidance](https://github.com/akanazawa/hmr/blob/master/README.md). 
After that, run the official demo script (`./HMR/demo.py`) to make sure HMR is properly setup. 

4. Setup **LIP** in `./LIP_JPPNet/` according to [this guidance](https://github.com/Engineering-Course/LIP_JPPNet/blob/master/README.md). 
After that, run the official demo script (`./LIP_JPPNet/evaluate_parsing_JPPNet-s2.py`) to make sure LIP is properly setup.  

5. If setup properly, the folder structure should look like:
```
    .
    ├── AlphaPose/
        ├── doc/
        ├── examples/
        ├── human-detection/
        ├── PoseFlow/
        └── ...
    ├── example/
    ├── hmr/
        ├── data/
        ├── doc/
        ├── models/
        ├── src/
        ├── __init__.py
        ├── demo.py
        └── ...
    ├── LIP_JPPNet/
        ├── checkpoint/
        ├── datasets/
        ├── kaffe/
        ├── utils/
        ├── __init__.py
        ├── evaluate_parsing_JPPNet-s2.py
        └── ...
    ├── smplify_public/
        ├── code/
        ├── README.md
        ├── requirements.txt
    ├── detect_bbox_by_parsing.py
    ├── detect_human.py
    ├── fit_3d_accurate.py
    ├── infer_smpl.py
    └── ...
```

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

### Results
(See `./example/` for more details. )
<p align="center">
<img src="./example/1.png.smpl_proj.png" title="SMPL reprojection" height="400", style="max-width:40%;vertical-align:top"> 
<img src="./example/1.png.meshlab.png" title="Output Mesh" height="400", style="max-width:40%;vertical-align:top"> 
</p>


### License
1. This code utilizes several open-source projects including 
AlphaPose(in ```./AlphaPose/```), HMR(in ```./hmr/```), LIP(in ```./LIP_JPPPNet/```) 
and SMPLify(in ```./smplify_public/```). 
They fall under [AlphaPose Liicense](https://github.com/MVIG-SJTU/AlphaPose/blob/master/LICENSE), 
[MIT License](https://github.com/akanazawa/hmr/blob/master/LICENSE), 
[MIT License](https://github.com/Engineering-Course/LIP_JPPNet/blob/master/LICENSE) and 
[SMPLIFY License](http://smplify.is.tue.mpg.de/data_license), respectively. 
By using this repository you agree to follow these licenses. 

2. The other part of the code falls under the following license:

> Copyright (c) 2019 Zerong Zheng, Tsinghua University
> 
> Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use Im2smpl software/data (the "Software"). By downloading and/or using the Software, you acknowledge that you have read these terms and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you must not download and/or use the Software.
> 
> **Ownership**
> 
> The Software has been developed at the Tsinghua University and is owned by and proprietary material of the Tsinghua University.
> 
> **License Grant**
> 
> Tsinghua University grants you a non-exclusive, non-transferable, free of charge right:
> 
> To download the Software and use it on computers owned, leased or otherwise controlled by you and/or your organisation;
> 
> To use the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.
> 
> Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, as training data for a commercial product, for commercial ergonomic analysis (e.g. product design, architectural design, etc.), or production of other artifacts for commercial purposes including, for example, web services, movies, television programs, mobile applications, or video games. The Software may not be used for pornographic purposes or to generate pornographic material whether commercial or not. This license also prohibits the use of the Software to train methods/algorithms/neural networks/etc. for commercial use of any kind. The Software may not be reproduced, modified and/or made available in any form to any third party without Tsinghua University’s prior written permission. By downloading the Software, you agree not to reverse engineer it.
> 
> **Disclaimer of Representations and Warranties**
> 
> You expressly acknowledge and agree that the Software results from basic research, is provided “AS IS”, may contain errors, and that any use of the Software is at your sole risk. TSINGHUA UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY KIND CONCERNING THE SOFTWARE, NEITHER EXPRESS NOR IMPLIED, AND THE ABSENCE OF ANY LEGAL OR ACTUAL DEFECTS, WHETHER DISCOVERABLE OR NOT. Specifically, and not to limit the foregoing, Tsinghua University makes no representations or warranties (i) regarding the merchantability or fitness for a particular purpose of the Software, (ii) that the use of the Software will not infringe any patents, copyrights or other intellectual property rights of a third party, and (iii) that the use of the Software will not cause any damage of any kind to you or a third party.
> 
> **Limitation of Liability**
> 
> Under no circumstances shall Tsinghua University be liable for any incidental, special, indirect or consequential damages arising out of or relating to this license, including but not limited to, any lost profits, business interruption, loss of programs or other data, or all other commercial damages or losses, even if advised of the possibility thereof.
> 
> **No Maintenance Services**
> 
> You understand and agree that Tsinghua University is under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. Tsinghua University nevertheless reserves the right to update, modify, or discontinue the Software at any time.
> 
> **Publication with the Software**
> 
> You agree to cite the paper describing the software and algorithm as specified on the download website.
> 
> **Media Projects with the Software**
> 
> When using the Software in a media project please give credit to Tsinghua University. For example: the Software was used for performance capture courtesy of the Tsinghua University.
> 
> **Commercial Licensing Opportunities**
> 
> For commercial use and commercial license please contact: liuyebin@mail.tsinghua.edu.cn.

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

@InProceedings{Zheng2019DeepHuman, 
    author = {Zheng, Zerong and Yu, Tao and Wei, Yixuan and Dai, Qionghai and Liu, Yebin},
    title = {DeepHuman: 3D Human Reconstruction From a Single Image},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```
