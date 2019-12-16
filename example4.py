import numpy as np

if __name__ == "__main__":
    a = np.arange(1, 10, 0.5)
    # print(a)
    # [1.  1.5 2.  2.5 3.  3.5 4.  4.5 5.  5.5 6.  6.5 7.  7.5 8.  8.5 9.  9.5]

    b = np.linspace(1, 10, 10)
    print(b)
    # [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]

    c = np.linspace(1, 10, 10, endpoint=False)
    print(c)
    # [1.  1.9 2.8 3.7 4.6 5.5 6.4 7.3 8.2 9.1]

    d = np.logspace(1, 2, 9, endpoint=True)
    print(d)
    # [ 10.          13.33521432  17.7827941   23.71373706  31.6227766
    # 42.16965034  56.23413252  74.98942093 100.        ]