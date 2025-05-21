# Adaptive Learning of Activation Functions via Genetic Algorithms

This project explores the use of genetic algorithms to evolve custom activation functions within multilayer perceptron (MLP) neural networks. Instead of relying on standard activation functions like sigmoid or ReLU, each network in the population learns its own activation function—represented as a discrete genome—through evolutionary processes such as selection, crossover, and mutation.

## 📘 Project Overview

- **Goal**: Automatically discover task-adaptive activation functions to improve neural network generalization.
- **Task**: A regression problem where the network learns to approximate a sine function over the interval \([-4, 4]\).
- **Method**:
  - Activation functions are encoded as a DNA (array of values), mapped to continuous functions via cubic interpolation.
  - Networks are trained for 10 epochs using backpropagation, and evaluated by Mean Squared Error (MSE).
  - Only the activation function evolves; network weights are randomly reinitialized each generation.

## 🧬 Genetic Algorithm Highlights

- **Population size**: 1000 (varied in some experiments)
- **Generations**: 30
- **Selection**: Tournament
- **Crossover**: Partial segment insertion (10% of DNA)
- **Mutation**: Flip Bit Mutation
- **Fitness**: Final training loss (MSE)

## 📂 Repository Structure

The following files compose the project:

### Main Script

- **`11)-Project-Genetic Algoritm.py`**  
  Main experiment pipeline. Initializes populations, trains networks, performs evolution, saves results, and visualizes performance.

### Modules

- **`Individual_class.py`**  
  Defines the `Individual` class representing a neural network with a custom activation function genome. Implements selection, crossover, mutation, and training logic.

- **`MLP_Custom_Class.py`**  
  Contains a custom PyTorch MLP model (`MLP_Custom`) that supports arbitrary activation functions generated from DNA.

- **`comparison_utils.py`**  
  Utilities for training and comparing networks, including baseline sigmoid model, plotting, and validation routines.

- **`utilities_func.py`**  
  Dataset generation, fitness tracking, plotting of activation functions, and saving/loading experiment data.

### Report

- **`Adaptive Learning of Activation Functions via Genetic Algorithms.pdf`**  
  The final report summarizing the project goals, methodology, experimental setup, results, and conclusions.

## 📊 Experiments Summary

Six main experiments (E1–E6) were conducted, each testing a different hypothesis:

- `E1` – Baseline configuration  
- `E2` – Extended codomain [0, 2]  
- `E3` – Short DNA (n = 6)  
- `E4` – Reduced population (p = 300)  
- `E5` – Larger network (h = 20)  
- `E6` – Symmetric codomain [-1, 1]

The best generalization performance was achieved in E6, with E3 and E2 also showing strong results. All evolved activation functions outperformed the baseline sigmoid model in validation error.

## 📈 Results and Takeaways

- Evolved activations can generalize better than fixed ones.
- Symmetric and expressive codomains improve flexibility.
- Genetic diversity enables the discovery of rare but highly effective solutions.
- Even with minimal training, the genetic algorithm successfully adapts activations to the task.

## ✅ Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

## 🚀 How to Run

1. Clone the repository.
2. Install dependencies.
3. Run `11)-Project-Genetic Algoritm.py` to launch the evolutionary experiments.
4. Use the plotting functions in `comparison_utils.py` to compare with baseline sigmoid models.

---

Created by Adele Caciolli, Caterina Cerretani, and Niccolò Petrilli – University of Siena  
