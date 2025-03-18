from compute_hessian import compute_hessian
from check_hessian_curvature import check_curvature
from plot_function import plot_surface_and_contours
import sympy as sp

def main():
    # Compute Hessian
    f, grad, hess, vars = compute_hessian()
    print("Function f(x, y):")
    print(f)
    print("\nGradient:")
    print(grad)
    print("\nHessian matrix:")
    sp.pprint(hess)

    # Check curvature at (0, 0)
    point = (0, 0)
    eigenvals, curvature = check_curvature(hess, vars, point)
    print(f"\nEigenvalues at point {point}: {eigenvals}")
    print(f"Curvature classification: {curvature}")

    # Plot surface and contours
    plot_surface_and_contours()

if __name__ == "__main__":
    main()
