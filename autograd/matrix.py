import numpy as np

import time
import multiprocessing as mp
from typing import Tuple, Callable

try: 
    from .primitives import Abs
    from .variable import Variable
    from .safety import ensure_mul, ensure_add, ensure_hadamard
except ModuleNotFoundError:
    from primitives import Abs
    from variable import Variable
    from safety import ensure_mul, ensure_add, ensure_hadamard


"""
For everything multiprocessing related that looks very esoteric, refer to this document:
    https://research.wmz.ninja/articles/2018/03/on-sharing-large-arrays-when-using-pythons-multiprocessing.html
"""
var_dict = {}

def rowmul(args):
    row_num, col_num = args
    row0, row1 = var_dict['mat0'][row_num], var_dict['mat1_t'][col_num]
    start_time = time.time()
    val = 0
    for i in range(len(row0)):
        val += row0[i] * row1[i]
    return val

def initialize_matrix(num_rows, num_cols, val):
    new = []
    for i in range(num_rows):
            row = [Variable(val) for _ in range(num_cols)]
            new.append(row)
    return Matrix(new)

def zeros(num_rows, num_cols):
    return initialize_matrix(num_rows, num_cols, 0)

class Matrix:
    def __init__(self, entries=[]):
        self.entries = entries

    def __str__(self):
        astr = '\n['
        for row in self.entries:
            astr += f'{str(row).replace(",", " ")}\n'
        astr = astr[:-1]
        astr += ']\n'
        return astr

    def __mul__(self, x):
        if isinstance(x, type(self)):
            if len(self) > 100:
                return self.matmul(x)
            return self.matmul_serial(x)
        return self.scalarmul(x)

    def __rmul__(self, x):
        """
        This could only ever happen if I tried to multiply a scalar
        by a matrix; if I did matrix matrix multiplication, __mul__ would have
        been called.
        """
        return self.scalarmul(x)

    def __truediv__(self, x):
        if isinstance(x, Matrix):
            print("Matrix - matrix division doesn't make sense, I think. Exiting...")
            raise TypeError
        return self.elementwise_apply(lambda y: y / x)

    def __add__(self, mat):
        ensure_add(self, mat)
        ans = zeros(len(self), len(self[0]))
        for row_num in range(len(self)):
            for col_num in range(len(self[0])):
                ans[row_num][col_num] = self[row_num][col_num] + mat[row_num][col_num]
        return ans

    def __sub__(self, mat):
        return self.__add__(mat.scalarmul(-1))

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.entries[idx]
        output = []
        for i in idx:
            output.append(self[i])
        return Matrix(output)

    def __setitem__(self, idx, val):
        self.entries[idx] = val

    def __len__(self):
        return len(self.entries)

    def __pow__(self, exponent):
        return self.elementwise_apply(lambda x: x**exponent)

    def transpose(self):
        """
        Implements the transpose operation, on self.entries.
        Does not do the transpose in place.
        Very cache inefficient.
        """
        old_entries = self.entries
        try:
            new = zeros(len(old_entries[0]), len(old_entries))
        except TypeError:
            new = zeros(1, len(old_entries))
        for row_num in range(len(old_entries)):
            try:
                for col_num in range(len(old_entries[0])):
                    new[col_num][row_num] = old_entries[row_num][col_num]
            except TypeError:
                new[0][row_num] = old_entries[row_num]
        return new

    def matmul_serial(self, mat):
        """
        Matrix, matrix multiplication. 
        Assumes that self and mat have the correct dimensionality.
        Serial version of matmul.
        """
        ensure_mul(self, mat)
        start_time = time.time()
        mat_t = mat.transpose()
        entries = zeros(len(self), len(mat[0]))
        for row_num in range(len(self)):
            start_row = time.time()
            for col_num in range(len(mat[0])):
                val = 0
                for entry_id in range(len(self[row_num])):
                    val += self[row_num][entry_id] * mat_t[col_num][entry_id]
                entries[row_num][col_num] = val
        return Matrix(entries)

    def matmul(self, mat):
        """
        Matrix, matrix multiplication.
        Assumes that mat0 and mat1 have the correct dimensionality.
        """
        ensure_mul(self, mat)
        mat_t = mat.transpose()
        entries = [None for _ in range(len(self))]

        def init_worker(mat0, mat1_t):
            var_dict['mat0'] = mat0 
            var_dict['mat1_t'] = mat1_t

        pool = mp.Pool(mp.cpu_count(), initializer=init_worker, initargs=(self, mat_t))

        for row_num in range(len(self)):
            start_time = time.time()
            row = pool.map(rowmul, [(row_num, col_num) for col_num in range(len(mat[0]))])
            entries[row_num] = list(row)

        return Matrix(entries)
            

    def scalarmul(self, c):
        """
        Scalar matrix multiplication.
        """
        new = self.zeros()
        for row_num in range(len(self)):
            for col_num in range(len(self[0])):
                new[row_num][col_num] = self[row_num][col_num]*c
        return new

    def hadamard(self, mat):
        """
        Elementwise multiplication with self and mat.
        """
        ensure_hadamard(self, mat)
        new = self.zeros()
        for row_num in range(len(self)):
            for col_num in range(len(self[0])):
                new[row_num][col_num] = self[row_num][col_num]*mat[row_num][col_num]
        return new

    def zeros(self):
        """
        Returns a new matrix of the same size as self filled with 0's.
        """
        return zeros(len(self), len(self[0]))

    def initialize_matrix(self, val):
        """
        Returns a new matrix of the same size as self filled with val.
        """
        return initialize_matrix(len(self), len(self[0]), val)

    def reset_grad(self):
        """
        Returns a new matrix, with reset Variable objects.
        """
        new = self.zeros()
        for row_num in range(len(self)):
            for col_num in range(len(self[0])):
                new[row_num][col_num] = Variable(self[row_num][col_num].value)
        return new
    
    def init_normal(self, mean=0, variance=1):
        """
        Fills its entries with samples from a normal distribution.
        """
        for row_num in range(len(self)):
            for col_num in range(len(self[0])):
                self[row_num][col_num] = Variable(np.random.normal(mean, variance))

    def get_grad(self, mat) -> float:
        """
        Obtains the gradient of self with respect to mat.
        For now, assumes self is of size 1x1.
        """
        if len(self) != 1 or len(self[0]) != 1:
            print('Autograd for matricies of shape != (1, 1) is currently not supported.')
            raise NotImplementedError

        grad = mat.zeros()
        for num_row in range(len(mat)):
            for num_col in range(len(mat[0])):
                grad[num_row][num_col] = self[0][0].grad_dict[id(mat[num_row][num_col])]
        return grad

    def elementwise_apply(self, f: Callable):
        """
        Assumes f is a function of one variable.
        Applies f to each entry in the matrix.
        """
        new = self.zeros()
        for num_row in range(len(self)):
            for num_col in range(len(self[0])):
                new[num_row][num_col] = f(self[num_row][num_col])
        return new

    def abs(self):
        new = self.zeros()
        for num_row in range(len(self)):
            for num_col in range(len(self[0])):
                new[num_row][num_col] = self[num_row][num_col].abs()
        return new

    def log(self):
        new = self.zeros()
        for num_row in range(len(self)):
            for num_col in range(len(self[0])):
                new[num_row][num_col] = self[num_row][num_col].log()
        return new

    def sum(self):
        """
        Returns the sum over every single element in self.
        """
        output = 0
        for num_row in range(len(self)):
            for num_col in range(len(self[0])):
                output += self[num_row][num_col]
        new = zeros(1, 1)
        new[0][0] = output
        return new

    def mean(self):
        """
        Returns the mean value over every single element in self.
        """
        return self.sum().elementwise_apply(lambda x: x/ (len(self) * len(self[0])))

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the matrix, as (num_rows, num_cols).
        """
        return (len(self), len(self[0]))

    def rowwise_apply(self, f: Callable):
        output = []
        for row in self:
            output.append([f(row)])
        return Matrix(output)

    def columnwise_apply(self, f: Callable):
        """
        Applies f to to each column in self.
        """
        mat = self.transpose()
        return mat.rowwise_apply(f).transpose()

    def max(self, axis: int):
        """
        :param axis: The axis for which we should get the maximum of.
        axis=0 corresponds to rows (ie, a column vector with the max of row i at entry i).
        axis=1 corresponds to columns (ie, a row vector with the max of column i at entry i).
        """
        return self.rowwise_apply(max) if axis == 0 else self.columnwise_apply(max)

    def min(self, axis: int):
        """
        :param axis: The axis for which we should get the minimum of.
        axis=0 corresponds to rows (ie, a column vector with the max of row i at entry i).
        axis=1 corresponds to columns (ie, a row vector with the max of column i at entry i).
        """
        return self.rowwise_apply(min) if axis == 0 else self.columnwise_apply(min)

    def append(self, row):
        """
        Appends the row to the matrix.
        """
        self.entries.append(row)


if __name__ == "__main__":
    mat = Matrix([[2, 2], [1, 1]])
    other_mat = Matrix([[1, 2, 3], [4, 5, 6]])
    inds = [0, 1]
    eval_list = [
            'mat * other_mat',
            'mat',
            'mat.transpose()',
            'mat * mat',
            'mat + mat',
            '10 * mat',
            'other_mat',
            'other_mat.max(axis=0)',
            'other_mat.max(axis=1)',
            'other_mat.min(axis=0)',
            'other_mat.min(axis=1)',
            'inds',
            'other_mat[inds]'
            ]

    for expr in eval_list:
        print(expr)
        print(eval(expr))

    def time_test(dim0, dim1, dim2, multi=False):
        import time

        A_vals = []
        for _ in range(dim0):
            A_vals.append(list(np.random.rand(dim1)))
        A = Matrix(A_vals)

        B_vals = []
        for _ in range(dim1):
            B_vals.append(list(np.random.rand(dim2)))
        B = Matrix(B_vals)

        start_time = time.time()
        if multi:
            A * B
        else:
            A.matmul_serial(B)
        print(time.time() - start_time)

    shape_list = [
            (100, 200, 150),
            (3, 5, 6),
            (10, 100, 40),
            (200, 10, 200),
            (500, 100, 200)
            ]

    for dim0, dim1, dim2 in shape_list:
        print(dim0, dim1, dim2)
        for multi in [True, False]:
            time_test(dim0, dim1, dim2, multi=multi)

