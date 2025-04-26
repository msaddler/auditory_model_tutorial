# Auditory modeling in PyTorch

This tutorial aims to introduce basic auditory modeling in PyTorch, a machine learning framework for the Python programming language. Some reasons to consider building auditory models in PyTorch include:

- **High-performance computing**: models can be efficiently run at scale on (multiple) CPUs or GPUs with minimal code modification.
- **Neural network compatibility**: models can be integrated with artificial neural networks, which are easily implemented in PyTorch.
- **Automatic differentiation**: neural network training typically relies on the **back propagation** algorithm, in which model parameters are optimized to minimize a loss function. This requires adjusting parameters according to the **gradient** of a loss function with respect to the given parameters. Machine learning frameworks like PyTorch have [**automatic differentiation**](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) built-in, meaning they automatically calculate and keep track of these gradients (greatly simplifying the code needed to train a network).

To illustrate some of these advantages, the [tutorial](TUTORIAL.ipynb) walks through:
1. Implementing a differentiable and GPU-compatible auditory nerve model
2. Crudely simulating peripheral effects of hearing loss
3. Defining a model-based loss function for hearing aid optimization
4. Optimizing hearing aid parameters via gradient descent


## Requirements

The [tutorial notebook](TUTORIAL.ipynb) was designed be run in [Google Colab](https://colab.research.google.com/), which requires a Google account. The Jupyter notebook can also be run locally with minimal Python 3.9+ [requirements](requirements.txt), though access to a CUDA-enabled GPU is highly recommended. The free-tier GPUs available through Google Colab are sufficient.

<a href="https://colab.research.google.com/github/msaddler/auditory_model_tutorial/blob/main/DEV.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## Contents

```
auditory_model_tutorial
|__ DEVELOPMENT.ipynb
|__ TUTORIAL.ipynb     <-- main tutorial notebook (start here)
|__ filters.py         <-- Python filter implementations
|__ modules.py         <-- PyTorch modules for auditory modeling
|__ utils.py           <-- helper functions for plotting and audio manipulation
|__ requirements.txt   <-- Python packages needed to run this code
|__ data               <-- toy dataset of 100 brief speech clips
    |__ 000.wav
    |__ 001.wav
        ...
    |__ 099.wav
```


## References

The provided here has been adapted from my own research:

- Models optimized for real-world tasks reveal the task-dependent necessity of precise temporal coding in hearing (Saddler & McDermott, 2024 Nature Communications): [paper](https://www.nature.com/articles/s41467-024-54700-5), [code](https://github.com/msaddler/phaselocknet)
- Speech denoising with auditory models (Saddler & Francl et al., 2021 Interspeech): [paper](https://arxiv.org/abs/2011.10706), [code](https://github.com/msaddler/auditory-model-denoising)


## Contact

Mark R. Saddler (marksa@dtu.dk)
