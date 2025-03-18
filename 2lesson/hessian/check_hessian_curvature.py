import sympy as sp

def check_curvature(hessian, variables, point):
    # Substitute the point in the Hessian
    substitutions = {var: val for var, val in zip(variables, point)}
    evaluated_hessian = hessian.subs(substitutions)
    
    # Compute eigenvalues
    eigenvals = evaluated_hessian.eigenvals()
    
    # Classify curvature
    eig_values_list = [float(val) for val in eigenvals.keys()]
    if all(ev > 0 for ev in eig_values_list):
        curvature_type = "Local minimum"
    elif all(ev < 0 for ev in eig_values_list):
        curvature_type = "Local maximum"
    else:
        curvature_type = "Saddle point"
    
    return eigenvals, curvature_type

if __name__ == "__main__":
    from compute_hessian import compute_hessian
    f, grad, hess, vars = compute_hessian()
    point = (0, 0)
    eigenvals, curvature = check_curvature(hess, vars, point)
    print(f"Eigenvalues at point {point}: {eigenvals}")
    print(f"Curvature classification: {curvature}")
