Softuni username: KamenPetrovv
Email: kamen.petrov.official@gmail.com
Discord: NoDramaLLama#8152

Project/Course github: https://github.com/KamenPetrovv/softuni-dl-2025
You can download the mlflow files from here if you want to explore the metrics yourself

The main project is described in: Project.ipynb

The last train session: Final train session.ipynb

A rough Gemini 3.1 generated script for quick testing: live_demo_attension.py

I left 3 folders:
Video_Sample -> You can download the youtube video from Amazon S3 bucket as described to test the download script (This uses the Video_Temp folder as well)
Processed_Video_Sample -> Use this to process the video into .pt tensors to test the preprocessing script if you want
Ava_ActiveSpeaker_Dataset_sample -> It contains a single .csv sample from the train set

Virtual envs:
- WSL/Linux for faster cuda preprocessing: envs/dl_env.yml
I was using this one during development for training, it has the cuda version for my 2060 super

- Windows for live demo: envs/active_speaker_demo_env.yml
I was using an env I had active_speaker, but I was using PowerShell, where `conda activate active_speaker` apperantly wasnt working, so I was installing things in either global or base env ... 
I have given the active_speaker_demo_env.yml from the base env. Not sure if it will work. 

