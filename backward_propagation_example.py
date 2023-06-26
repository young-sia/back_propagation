import numpy as np
import matplotlib


def nonlinear(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


def main():
    train33 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
    y = np.array([[0, 0, 1, 1]]).T
    # np.random.seed(1)
    syn0 = np.random.random((3, 1))
    # syn1 = 2 * np.random.random((4, 1)) - 1
    for j in range(10000):
        l0 = train33
        l1 = nonlinear(np.dot(l0, syn0))

        l1_error = y - l1
        l1_delta = l1_error * nonlinear(l1, True)
        syn0 += np.dot(l0.T, l1_delta)
        print("output after training")
        print(l1)



if __name__ == '__main__':
    main()
