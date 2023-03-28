# Create VM instance

1. Create VM instance with the following specs:
   - Location/zone: europe-west4-a
   - GPU: NVIDEA Tesla P100
   - Activate CUDA cores
     - In same menu change disk size to 250Gb and select SSD
   - Check the both HTTPS boxes
2. After loading into your VM by clicking SSH
    - Will probably install drivers automatically

# Download model and data
1. Download dataset:
```
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip
```
2. Upload labels manually
3. Download Calibration:
```
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip
```
4. Download pre-trained model:
```
pip install gdown
gdown --id 11mUMdFPNT-05lC54Ru_2OwdwqTPV4jr
```
5. Make directory:\
```mkdir data```
6. Unzip data in folders in data directory: 
```
unzip -q data_odometry_data.zip -d data
unzip -q data_odometry_labels.zip -d data
unzip -q data_odometry_calib.zip -d data
```
7. Remove zip files:
```
rm -r data_odometry_data.zip
rm -r data_odometry_labels.zip
rm -r data_odometry_calib.zip
```
8. Make output directory for storing the results
```
mkdir output
```

# Setup Conda
1. Upload conda environment.yml
2. Create conda environment
```
conda env create -f environment.yml
```
3. Activate environment
```
conda activate DL_project
```

# Setup CUDA
1. Check CUDA version on the top right of the given output (probably 11.6)
```
nvidia-smi
```
2. Setup torch to use this version of cuda (this could be different for other cuda versions, see https://discuss.pytorch.org/t/problem-with-cuda-11-6-installation/156119)
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

# Test the program
1. Check all the paths in the python program
2. Run the program (if correct conda environment is activated)
```
python ./run_inference.py
```
3. Check if no errors occur

# Run the program as background process
To run the program, even if you closed the VM, you need to run the script as a background process and save the outputs in a text file.
1. Get location of python version
```commandline
which python
```
2. Start the background process
```commandline
nohup <absolute path to anaconda python> -u <absolute path to python script> > outputfile.txt &
```
3. Check if the outputfile.txt contains the output of the program, if not something went wrong
```commandline
cat outputfile.txt
```
4. Check if the process is running in the background, you should see multiple running process
```commandline
ps aux | grep <Name of running script>
```
5. If you want to check the status of your program at a later time
```commandline
cat outputfile.txt
```

# Download results
If the program is finished, download the results
1. ZIP the output folder (if line is not working, delete -q)
```commandline
zip -r -q results.zip output
```
2. Download the results.zip from the VM