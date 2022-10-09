import Tema1_utils
import numpy as np

(coef, res) = Tema1_utils.read_equations()

coef = np.array(coef)
res = np.array(res)

print("Matrix of coeficients is:")
Tema1_utils.print_matrix(coef)

print("Matrix of results is:")
Tema1_utils.print_matrix(res)

vars = np.linalg.solve(coef, res)

print("Solution: ")
print("Var x: ", round(vars[0]))
print("Var y: ", round(vars[1]))
print("Var z: ", round(vars[2]))

print("Check valid solution: ", np.allclose(np.dot(coef, vars), res))