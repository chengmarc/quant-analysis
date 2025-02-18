# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:20:15 2025

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt


def plot(X, Y, Z, title='3D Explicit Surface Visualization'):    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.9)
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    plt.show()


# %% Example Usage

# Define domain
x = np.linspace(-1, 0.99, 100)
y = np.linspace(-1, 0.99, 100)
X, Y = np.meshgrid(x, y)

# Define explicit surface on codomain
Z = np.floor(X) + np.floor(Y) + 0.5 * (np.abs(X % 1 - 0.5) + np.abs(Y % 1 - 0.5))

plot(X, Y, Z, 'Gradient Visualization')

