## Hockey Penalty Classification

Steps to train model -

1. Define path to SBU Kinect dataset in `datasets/sbu/train_test_split.py`. 
2. Define model hyperparameters in configs/dataset/sbu.yaml. Important ones are batch_size, num_epochs, learning_rate and model (model can be either 'model1'/'model2'/'model3'/'model4'). For the model chosen, update its corresponding hyperparameters in the same config file.
3. Run `python train.py`