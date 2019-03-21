from .safety import ensure_mul, ensure_add


def zeros(num_rows, num_cols):
    row = [0] * num_cols
    new = []
    for i in range(num_rows):
            new.append(row.copy())
    return Matrix(new)

class Matrix:
    def __init__(self, entries):
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
            ensure_mul(self, x)
            return self.matmul(self, x)
        return self.scalarmul(x)

    def __rmul__(self, x):
        """
        This could only ever happen if I tried to multiply a scalar
        by a matrix; if I did matrix matrix multiplication, __mul__ would have
        been called.
        """
        return self.scalarmul(x)

    def __add__(self, mat):
        ensure_add(self, mat)
        ans = zeros(len(self), len(self[0]))
        for row_num in range(len(self)):
            for col_num in range(len(self[0])):
                ans[row_num][col_num] = self[row_num][col_num] + mat[row_num][col_num]
        return ans

    def __getitem__(self, idx):
        return self.entries[idx]

    def __len__(self):
        return len(self.entries)

    def transpose(self):
        """
        Implements the transpose operation, on self.entries.
        Does not do the transpose in place.
        Very cache inefficient.
        """
        old_entries = self.entries
        new = zeros(len(old_entries[0]), len(old_entries))
        for row_num in range(len(old_entries)):
            for col_num in range(len(old_entries[0])):
                new[col_num][row_num] = old_entries[row_num][col_num]
        return new

    def matmul(self, mat0, mat1):
        """
        Matrix, matrix multiplication. 
        Assumes that mat0 and mat1 have the correct dimensionality.
        """
        #ensure_correct_shape_for_multiplication(mat0, mat1)
        mat1_t = mat1.transpose()
        entries = zeros(len(mat0), len(mat1[0]))
        for row_num in range(len(mat0)):
            for col_num in range(len(mat1[0])):
                val = 0
                for entry_id in range(len(mat0[row_num])):
                    val += mat0[row_num][entry_id] * mat1_t[col_num][entry_id]
                entries[row_num][col_num] = val
        return entries

    def scalarmul(self, c):
        """
        Scalar matrix multiplication.
        """
        new = zeros(len(self), len(self[0]))
        for row_num in range(len(self)):
            for col_num in range(len(self[0])):
                new[row_num][col_num] = self[row_num][col_num]*c
        return new

    def zeros(self):
        """
        Returns a new matrix of the same size as self filled with 0's.
        """
        return zeros(len(self), len(self[0]))

if __name__ == "__main__":
    mat = Matrix([[2, 2], [1, 1]])
    mat_t = mat.transpose()
    print(mat)
    print(mat_t)
    mul = mat * mat
    print(mul)
    add = mat + mat
    print(add)
    other_mat = Matrix([[1, 2, 3], [4, 5, 6]])
    print(mat * other_mat)
    print(10 * mat)
