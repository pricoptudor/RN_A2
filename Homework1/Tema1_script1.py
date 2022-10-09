import Tema1_utils

(coef, res) = Tema1_utils.read_equations()

print("Matrix of coeficients is:")
Tema1_utils.print_matrix(coef)

print("Matrix of results is:")
Tema1_utils.print_matrix(res)

# Matrix coef  *  matrix [x,y,z] = matrix res
# => Matrix [x,y,z] = matrix coef ^ -1  *  matrix res

vars = Tema1_utils.multiply_matrix_3x3_3x1(Tema1_utils.matrix_inverse(coef), res)

print("Solution: ")
print("Var x: ", round(vars[0]))
print("Var y: ", round(vars[1]))
print("Var z: ", round(vars[2]))
