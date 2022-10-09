### File ops:

def parse_equation(equation):
    #compute result of equation:
    equal_index = equation.index('=')
    r = int(equation[equal_index+1:])
    # print(equal_index, r)

    #compute coeficients of equation:
    try:
        x_index = equation.index('x')

        if equation[:x_index] == '+':
            a = 1
        elif equation[:x_index] == '-':
            a = -1
        else:
            a = int(equation[:x_index])

        # print(x_index, a)
    except ValueError:
        a = 0
        x_index = 0

    try:
        y_index = equation.index('y')

        if equation[x_index+1:y_index] == '+':
            b = 1
        elif equation[x_index+1:y_index] == '-':
            b = -1
        else:
            b = int(equation[x_index+1:y_index])
        
        # print(y_index, b)
    except ValueError:
        b = 0
        y_index = x_index

    try:
        z_index = equation.index('z')

        if equation[y_index+1:z_index] == '+':
            c = 1
        elif equation[y_index+1:z_index] == '-':
            c = -1
        else:
            c = int(equation[y_index+1:z_index])
        
        # print(z_index, c)
    except ValueError:
        c = 0
        z_index = y_index

    return (a,b,c,r)

def clear_equation(equation):
    if equation[len(equation)-1] == '\n':
        equation = equation[:-1]
    equation.replace(' ','')
    return equation

def read_equations():
    file = open("input_file", "r")

    (a1,b1,c1,r1) = parse_equation(clear_equation(file.readline()))
    (a2,b2,c2,r2) = parse_equation(clear_equation(file.readline()))
    (a3,b3,c3,r3) = parse_equation(clear_equation(file.readline()))

    file.close()

    return (((a1,b1,c1),(a2,b2,c2),(a3,b3,c3)),(r1,r2,r3))


### Matrix calculus (for 3x3):

def matrix_determinant_3x3(matrix):
    det = matrix[0][0] * matrix[1][1] * matrix[2][2] +\
            matrix[1][0] * matrix[2][1] * matrix[0][2] +\
                matrix[0][1] * matrix[1][2] * matrix[2][0]
    det = det -\
            matrix[0][2] * matrix[1][1] * matrix[2][0] -\
                matrix[0][1] * matrix[1][0] * matrix[2][2] -\
                    matrix[1][2] * matrix[2][1] * matrix[0][0]
    return det

def matrix_transpose(matrix):
    transpose = [[0] * 3 for i in range(3)]
    for i in range(3):
        for j in range(3):
            transpose[i][j] = matrix[j][i]
    return transpose

### first second
### third fourth
def get_small_matrix_determinant_for_adj(matrix, x, y):
    terms = []
    for i in range(3):
        for j in range(3):
            if i!=x and j!=y:
                terms.append(matrix[i][j])
    return terms[0] * terms[3] - terms[1] * terms[2]


def matrix_adjugate(matrix):
    transpose = matrix_transpose(matrix)
    adj = [[0] * 3 for i in range(3)]

    for i in range(3):
        for j in range(3):
            adj[i][j] = (-1)**(i+j) *\
                 get_small_matrix_determinant_for_adj(transpose,i,j)

    return adj

def matrix_inverse(matrix):
    matrix_det = matrix_determinant_3x3(matrix)

    if(matrix_det == 0):
        print("The matrix determinant is 0 hence the matrix is not inversable")
    else:
        matrix_adj = matrix_adjugate(matrix)
        for i in range(3):
            for j in range(3):
                matrix_adj[i][j] = matrix_adj[i][j] / matrix_det 
        print("The inverse of the matrix is: ")
        print_matrix(matrix_adj)
        return matrix_adj

def multiply_matrix_3x3_3x3(matrix1, matrix2):
    matrix3 = [[0] * 3 for i in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                matrix3[i][j] = matrix3[i][j] + matrix1[i][k] * matrix2[k][j]
    return matrix3

def multiply_matrix_3x3_3x1(matrix1, matrix2):
    matrix3 = [0] * 3
    for i in range(3):
        for j in range(3):
            matrix3[i] = matrix3[i] + matrix1[i][j] * matrix2[j]
    return matrix3

def print_matrix(matrix):
    for i in range(3):
        print(matrix[i])

