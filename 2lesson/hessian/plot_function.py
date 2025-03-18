import numpy as np
import matplotlib.pyplot as plt

def func(x, y):
    return x**2 + x*y + y**2

def plot_surface_and_contours():
    # Create mesh grid
    X = np.linspace(-3, 3, 100)
    Y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(X, Y)
    Z = func(X, Y)
    
    fig = plt.figure(figsize=(12, 6))

    # 3D Surface Plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.9)
    ax1.set_title('Surface of f(x, y)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x, y)')

    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=1, fontsize=8)
    ax2.set_title('Contours of f(x, y)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_surface_and_contours()
