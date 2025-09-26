import numpy as np

# Define space
x = np.linspace(-3.14, 3.14, 1000)   # symmetric range, fine resolution
dx = x[1] - x[0]               # spacing between points

# Define your kernel function
def kernel_cos(x, A1, A2, a1, a2):
    return (A1 * np.exp(-a1 * x**2) - A2 * np.exp(-a2 * x**2)) * np.cos(x / 2)

# Choose parameters
A1, A2, a1, a2 = 6, 1, 30, 0.9
kernel = kernel_cos(x, A1, A2, a1, a2)

# Compute the numerical integral using the trapezoidal rule
integral = np.trapz(kernel, x)   # or use sum(kernel)*dx for a rougher estimate
print(f"Kernel integral: {integral:.6f}")