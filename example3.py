import numpy as np

if __name__ == "__main__":
    a = np.arange(0, 60, 10).reshape((-1, 1)) + np.arange(6)
    # print(a)
    # [[ 0  1  2  3  4  5]
    # [10 11 12 13 14 15]
    # [20 21 22 23 24 25]
    # [30 31 32 33 34 35]
    # [40 41 42 43 44 45]
    # [50 51 52 53 54 55]]

    # print(a.dtype)
    # int32

    l = [1, 2, 3]
    # print(l)
    # [1, 2, 3]

    la = np.array(l)
    # print(la)
    # [1 2 3]

    # print(type(la), type(l))
    # <class 'numpy.ndarray'> <class 'list'>

    b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    # print(b)
    # [[1 2 3]
    # [4 5 6]
    # [7 8 9]
    # [10 11 12]]

    # print(a.shape)
    # (6, 6)
    # print(b.shape)
    # (4, 3)

    b.shape = (3, 4)
    # print(b)
    # [[ 1  2  3  4]
    # [ 5  6  7  8]
    # [ 9 10 11 12]]

    c = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float)
    d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.complex)
    # print(c)
    # print(d)
    # [[ 1.  2.  3.  4.]
    # [ 5.  6.  7.  8.]
    # [ 9. 10. 11. 12.]]
    # [[ 1.+0.j  2.+0.j  3.+0.j  4.+0.j]
    # [ 5.+0.j  6.+0.j  7.+0.j  8.+0.j]
    # [ 9.+0.j 10.+0.j 11.+0.j 12.+0.j]]

    e = c.astype(np.int)
    # print(e)
    # [[ 1  2  3  4]
    # [ 5  6  7  8]
    # [ 9 10 11 12]]