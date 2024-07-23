# A Bioacoustics Fork of the Noise2Noise Audio Denoising Repository
Code in this repository is based on code published alongside the following paper: "Speech Denoising without Clean Training Data: a Noise2Noise Approach". 
It has been adapted for marine bioacoustics purposes, and thus has been trained on ocean noise recorded on hydrophones. The salient signal embedded in the noise is a whale call.

## Research Paper and Citation
 https://www.isca-speech.org/archive/interspeech_2021/kashyap21_interspeech.html 
[Arxiv](https://arxiv.org/abs/2104.03838). 

Cite the original work:

@inproceedings{kashyap21_interspeech,\
  author={Madhav Mahesh Kashyap and Anuj Tambwekar and Krishnamoorthy Manohara and S. Natarajan},\
  title={{Speech Denoising Without Clean Training Data: A Noise2Noise Approach}},\
  year=2021,\
  booktitle={Proc. Interspeech 2021},\
  pages={2716--2720},\
  doi={10.21437/Interspeech.2021-1130}\
}

## Python Requirements - MODIFIED FROM ORIGINAL REPOSITORY
The original project utilises a number of outdated libraries. An environment manager is therefore necessary.
The original project uses a requirements.txt to specify packages to Pip, however some are no longer found on Conda channels.
The following instructions work around this. They assume a clean system, and are known-good as of 23/07/2024. 

## Method 1 - Modified version of original developer's "requirements.txt"
1 - Install Anaconda Navigator.\
2 - Install Anaconda Prompt (from inside Anaconda Navigator, base env activated).\
3 - Launch Anaconda Prompt (from inside Anaconda Navigator, base env activated).\
4 - Navigate to noise2noise repo directory - ```cd user/gitroot/Noise2Noise_Audio```, replacing "user", "git" with your actual path's folder names.\
5 - Create a new environment ```conda create --name env --file requirements.txt```, replacing "env" with a name for your new environment.\
6 - The packages that no longer exist on conda channels are commented out, at the top. These must be installed manually.\
7 - In Anaconda Prompt, activate env ```conda activate env``` where "env" is your chosen env name.\
8 - Install pip to your conda env with ```conda install pip```.\
9 - Still in Anaconda Prompt, navigate to your conda environment's folder, which should be something like ```user/anaconda3/envs/env/```.\
10 - Run ```pip install package_name``` where package name is the name of the first commented out package at the top of the list in requirements.txt.\
11 - Repeat step 10 for the other packages.\
12 - Install Anaconda Toolbox.\
13 - Install JupyterLab to run notebooks.\

## Method 2 - This project's YAML file
1 - Install Anaconda Navigator.\
2 - Go to "environments", select "Import environment", select ```N2N_Audio_CondaEnvBackup.yaml``` from this git repo.
12 - Install Anaconda Toolbox.\
13 - Install JupyterLab to run notebooks.\

## Dataset Generation
We use 2 standard datasets; 'UrbanSound8K'(for real-world noise samples), and 'Voice Bank + DEMAND'(for speech samples). 
Please download the datasets from [urbansounddataset.weebly.com/urbansound8k.html](https://urbansounddataset.weebly.com/urbansound8k.html) 
and [datashare.ed.ac.uk/handle/10283/2791](https://datashare.ed.ac.uk/handle/10283/2791) respectively. 
Extract and organize into the Datasets folder as shown below:

```
Noise2Noise-audio_denoising_without_clean_training_data
│     README.md
│     speech_denoiser_DCUNet.ipynb
|     ...
│_____Datasets
      |     clean_testset_wav
      |     clean_trainset_28spk_wav
      |     noisy_testset_wav
      |     noisy_trainset_28spk_wav
      |_____UrbanSound8K
            |_____audio
                  |_____fold1
                  ...
                  |_____fold10

```

To train a White noise denoising model, run the script:
```
python white_noise_dataset_generator.py
```

To train a UrbanSound noise class denoising model, run the script, and select the noise class:
```
python urban_sound_noise_dataset_generator.py

0 : air_conditioner
1 : car_horn
2 : children_playing
3 : dog_bark
4 : drilling
5 : engine_idling
6 : gun_shot
7 : jackhammer
8 : siren
9 : street_music
```
The train and test datasets for the specified noise will be generated in the 'Datasets' directory.

## Training a New Model
In the 'speech_denoiser_DCUNet.ipynb' file. Specify the type of noise model you want to train to denoise
(You have to generate the specific noise Dataset first). 
You can choose whether to train using our Noise2Noise approach(using noisy audio for both training inputs and targets), 
or the conventional approach(using noisy audio as training inputs and the clean audio as training target). 
If you are using Windows, set 'soundfile' as the torchaudio backend. If you are using Linux, set 'sox' as the torchaudio backend. 
The weights .pth file is saved for each training epoch in the 'Weights' directory.

## Testing Model Inference on Pretrained Weights
We have trained our model with both the Noise2Noise and Noise2Clean approaches, for all 10(numbered 0-9) 
UrbanSound noise classes and White Gaussian noise. All of our pre-trained model weights are uploaded in 'Pretrained_Weights' 
directory under the 'Noise2Noise' and 'Noise2Clean' subdirectories.

In the 'speech_denoiser_DCUNet.ipynb' file. Select the weights .pth file for model to use. 
Point to the testing folders containing the audio you want to denoise. 
Audio quality metrics will also be calculated. 
The noisy, clean and denoised wav files will be saved in the 'Samples' directory.

## Example
### Noisy audio waveform
![./static/noisy_waveform.PNG](./static/noisy_waveform.PNG)
### Model denoised audio waveform
![static/denoised_waveform.PNG](static/denoised_waveform.PNG)
### True clean audio waveform
![static/clean_waveform.PNG](static/clean_waveform.PNG)
## 20-Layered Deep Complex U-Net 20 Model Used
![static/dcunet20.PNG](static/dcunet20.PNG)
## Results
![static/results.PNG](static/results.PNG)
