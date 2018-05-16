# Learning Heuristic Search via Imitation
**********

Official repository containing OpenAI Gym environments, agents and ML models for the CoRL paper [Learning Heuristic Search via Imitation](https://arxiv.org/pdf/1707.03034.pdf)

# External Dependencies
1. [OpenAI gym](https://gym.openai.com/envs/)
2. [Numpy](http://www.numpy.org/)
3. [Matplotlib](https://matplotlib.org/)
4. [TensorFlow](https://www.tensorflow.org/)
5. [tflearn](http://tflearn.org/installation/)

# Getting Started
One you have installed the required external dependencies (favorably in a virtualenv), you need to execute the following steps in order to get started with the examples.

 - Create a meta folder for the project ``mkdir ~/heuristic_learning `` 
 - Get the 2D planning datasets: ``git clone git@github.com:mohakbhardwaj/motion_planning_datasets.git``
 - Get the search based planning backend: ``git clone git@github.com:mohakbhardwaj/planning_python.git``
 - Get the SaIL repository (this repo): ``git clone git@github.com:mohakbhardwaj/SaIL.git``
 - Go to the ``examples/`` folder: ``cd ~/heuristic_learning/SaIL/SaIL/examples``
 - Run ``./run_generate_oracles_xy.sh`` which will generate oracles for all the train, validation and test datasets in the ``motion_planning_datasets`` folder
 - Run ``./run_sail_xy_train.sh`` to train a heuristic for one of the datasets (you can specify the dataset you want inside the script). This runs SaIL for 10 iterations by default. For more information on the rest of the parameters used see the file ``sail_xy_train.py`` 

# Contact
For more information contact
Mohak Bhardwaj : [mohak.bhardwaj@gmail.com](mohak.bhardwaj@gmail.com)