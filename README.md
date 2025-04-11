# Tutorial: auditory modeling in PyTorch

This tutorial aims to introduce basic auditory modeling in PyTorch, a machine learning framework for the Python programming language. Some reasons to consider building auditory models in PyTorch include:

- **High-performance computing**: models can be efficiently run at scale on (multiple) CPUs or GPUs with minimal code modification.
- **Neural network compatibility**: models can be integrated with artificial neural networks, which are easily implemented in PyTorch.
- **Automatic differentiation**: neural network training typically relies on the **back propagation** algorithm, in which model parameters are optimized to minimize a loss function. This requires adjusting parameters according to the **gradient** of a loss function with respect to the given parameters. Machine learning frameworks like PyTorch have [**automatic differentiation**](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) built-in, meaning they automatically calculate and keep track of these gradients (greatly simplifying the code needed to train a network).

## Setup
Please (attempt to) complete these steps before the tutorial

1. **Install a code text editor with interactive support for Jupyter Notebooks**: I suggest [VSCode](https://code.visualstudio.com/).
2. **Install Python and PyTorch**: the details will depend on your operating system, and instructions for different systems can be found here https://pytorch.org/get-started/locally/. These are the steps I took to install the required packages in a new Conda environment on MacOS:
```
$ conda create --name=tutorial python=3.12
$ conda activate tutorial
$ conda install pip
$ pip3 install torch torchvision torchaudio
$ pip3 install notebook matplotlib scipy
```
