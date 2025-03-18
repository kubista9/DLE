import sympy as sp

def compute_hessian():
    # Define symbols
    x, y = sp.symbols('x y')
    
    # Define the function
    f = x**2 + x*y + y**2
    
    # Compute gradient (first derivatives)
    gradient = [sp.diff(f, var) for var in (x, y)]
    
    # Compute Hessian matrix (second derivatives)
    hessian = sp.hessian(f, (x, y))
    
    return f, gradient, hessian, (x, y)

if __name__ == "__main__":
    f, grad, hess, vars = compute_hessian()
    print("Function:")
    print(f)
    print("\nGradient:")
    print(grad)
    print("\nHessian:")
    sp.pprint(hess)
