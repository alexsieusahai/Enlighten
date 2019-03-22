def ensure_mul(mat0, mat1):
    """
    Ensures that mat0 and mat1 are the proper shapes for multiplicaton.
    """
    if len(mat0.entries[0]) != len(mat1.entries):
        print(f'Matrix sizes do not agree for multiplication:\n \
                {len(mat0.entries[0])} != {len(mat1.entries)}')
        raise TypeError

def ensure_add(mat0, mat1):
    """
    Ensures that mat0 and mat1 are the proper shapes for addition.
    """
    if len(mat0) != len(mat1):
        print(f'Matrix sizes do not agree for addition in dimension 0: \n \
                {len(mat0)} != {len(mat1)}')
        raise TypeError
    if len(mat0[0]) != len(mat1[0]):
        print(f'Matrix sizes do not agree for addition in dimension 1: \n \
                {len(mat0[0])} != {len(mat1[0])}')
        raise TypeError
