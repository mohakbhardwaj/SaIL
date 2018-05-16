# Learning Heuristic Search via Imitation
**********

Official repository containing OpenAI Gym environments, agents and ML models for the CoRL paper [Learning Heuristic Search via Imitation](https://arxiv.org/pdf/1707.03034.pdf)

# External Dependencies
1. [OpenAI gym](https://gym.openai.com/envs/)
2. [Numpy](http://www.numpy.org/)
3. [Matplotlib](https://matplotlib.org/)
4. [Tensorflow](https://www.tensorflow.org/)
5. [tflearn](https://http://tflearn.org/installation/)

# Setting Up
One you have installed the required external dependencies (favorably in a virtualenv), you need to execute the following steps in order to get started with the examples.

 - Create a meta folder for the project ``mkdir ~/heuristic_learning `` 
 - Get the 2D planning datasets: `` git clone git@github.com:mohakbhardwaj/motion_planning_datasets.git ``
 - Get the search based planning backend: `` git clone git@github.com:mohakbhardwaj/planning_python.git ``
 - Get the SaIL repository (this repo): ``git clone git@github.com:mohakbhardwaj/SaIL.git ``

Search based planning pipeline: [mohakbhardwaj/planning_python](https://github.com/mohakbhardwaj/planning_python)
See ``examples/`` folder for basic examples of generating oracles, training SaIL on different 2D databases and testing the learnt heuristic.
		


