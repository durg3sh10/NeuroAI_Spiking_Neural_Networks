# NeuroAI_Spiking_Neural_Networks

ES 673 NeuroAI, Assignment 4 (Q2): a feedforward spiking neural network in Brian2 on Poisson-encoded MNIST.

- 784 Poisson inputs (100 ms per image, max 100 Hz) -> 400 LIF hidden neurons -> logistic regression on spike counts
- Input weights are fixed (N(0.5, 0.1), clipped, connection probability 0.2); only the readout is trained
- Tweaks: homeostatic firing-rate regulation (adaptive threshold), membrane time constant sweep, sparse connectivity

## Results (6000 train / 1000 test images)

| Experiment | Setting | Test acc | Spikes / image |
|---|---|---|---|
| Baseline | p = 0.2, tau_m = 20 ms | 0.824 | 1615 |
| Homeostasis | target 20 Hz | 0.791 | 795 |
| Homeostasis | target 40 Hz | 0.827 | 1560 |
| tau_m | 5 ms | 0.290 | 27 |
| tau_m | 10 ms | 0.761 | 660 |
| tau_m | 50 ms | 0.836 | 2186 |
| Connectivity | p = 1.0 (dense) | 0.563 | 1570 |
| Connectivity | p = 0.05 | 0.839 | 1676 |
| Reference | logistic regression on raw pixels | 0.873 | - |

## Running

```
pip install brian2 torch torchvision scikit-learn matplotlib
jupyter notebook assignment4_q2.ipynb
```

MNIST is downloaded to `./data` on the first run. The full notebook takes about 40 minutes on CPU.
