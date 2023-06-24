# BCDA

<img src="model_architecture.png" width="500" />


## Usage
Pretraining for the Voice Activity Projection Model:
```
>>> python vap_train.py data/conf/vap-best.yaml
```

Training for the Backchannel Prediction Model:
```
>>> python bcda_train.py data/conf/whole.yaml --pretrained data/model/vap_pretrained_model-best.ckpt
```
All configurations used in the paper are found in the folder `data/conf`.



## Set Up

#### Step 0:
Create and activate a new clean conda environment:
```
>>> conda create -n myenv python=3.9
>>> conda activate myenv
```

#### Step 1:
In order to install the appropraite pytorch version, first find your CUDA version:
In Windows Powershell or Linux standard terminal:
```
>>> nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.65.01    Driver Version: 516.94       CUDA Version: **11.7**     |
|-------------------------------+----------------------+----------------------+
...
```

#### Step 2:
Find your matching pytorch version and copy the comand from:
https://pytorch.org/get-started/previous-versions/
There is no version for CUDA 11.7 listed, but we make use of backwards compatibility and install:
```
>>> conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

#### Step 3
Install all standard dependencies:
```
>>> pip install -r requirements.txt
```

#### Step 4:
Manually install the contrastive predictive wave encoding model:
```
>>> pip install git+https://github.com/facebookresearch/CPC_audio.git
```
The download may fail and ask you to install the C++ development tool with Visual Studio. Do so. Afterwards execute the above command again.

#### Step 5:
Prepare the data. You will need the freely available transcripts as well as the licenced audio data.  
Download transcripts from: https://www.openslr.org/resources/5/switchboard_word_alignments.tar.gz  
Buy audio files from: https://catalog.ldc.upenn.edu/LDC97S62  
Finally, place all data into the subfolder `data/swb`. The folder structure should look like this:

data/swb/  
&nbsp;&nbsp;-> swb_audios   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> sw02001.sph  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> sw02005.sph  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> ...  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> sw04940.sph  
&nbsp;&nbsp;-> swb_ms98_transcriptions  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> 20/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> 2001/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> ...  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> 21/  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> ...  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-> 49/  
