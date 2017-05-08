import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import copy
import numpy.linalg as linalg
#import elice_utils
import math


def read_data(filename):
    X = []
    # Read the dataset here...

    with open(filename) as fp:
        N = int(fp.readline())
        for line_idx in range(N):
            x_i = [float(x) for x in fp.readline().strip().split()]
            X.append(x_i)

    # X must be the N * 2 numpy array.
    X = np.array(X)
    return X


def gaussian(mu, sigma, x):
    # Use this function to get the density of multivariate normal distribution
    from scipy.stats import multivariate_normal
    var = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)

    # add an extremely small probability to avoid zero probability
    return var.pdf(x) + 10 ** -20


def get_initial_random_state(X, K):
    import random
    random.seed(0)

    X1 = X[:, 0]
    X2 = X[:, 1]

    mu = []
    sigma = []
    pi = []

    for k in range(K):
        x1 = random.uniform(min(X1), max(X1))
        x2 = random.uniform(min(X2), max(X2))
        mu.append([x1, x2])
        sigma.append([[1, 0],
                      [0, 1]])
        pi.append(1 / K)

    mu = np.array(mu)
    sigma = np.array(sigma)
    pi = np.array(pi)
    return (mu, sigma, pi)


def kmeans(X, theta):
    mu, sigma, pi = theta
    K = len(mu)

    labels = np.zeros(len(X))

    for i in range(len(X)):
        d = []
        for k in range(K):
            d.append(np.sqrt((X[i][0] - mu[k][0]) ** 2 + (X[i][1] - mu[k][1]) ** 2))
        labels[i] = np.argmin(d)
    while True:
        for k in range(K):
            x1 = 0.0
            x2 = 0.0
            cnt = 0
            for i in range(len(X)):
                if labels[i] == k:
                    x1 += X[i][0]
                    x2 += X[i][1]
                    cnt += 1
            mu[k][0] = x1 / float(cnt)
            mu[k][1] = x2 / float(cnt)

        labels_old = copy.deepcopy(labels)

        for i in range(len(X)):
            d = []
            for k in range(K):
                d.append(np.sqrt((X[i][0]-mu[k][0])**2 + (X[i][1]-mu[k][1])**2))
            labels[i] = np.argmin(d)

        if np.array_equal(labels, labels_old):
            break

    return (mu, sigma, pi)


def expected_complete_LL(X, R, K, theta):
    ll = 0
    mu, sigma, pi = theta

    for i in range(len(X)):
        tmp = 0.0
        for k in range(K):
            tmp += R[i][k]*np.log(pi[k]) + R[i][k]*np.log(gaussian(mu[k], sigma[k], X[i]))
        ll += tmp

    return ll


def expect(X, theta):
    # unpack
    mu, sigma, pi = theta
    R = []

    K = len(mu)

    for i in range(len(X)):
        tmp = []
        for k in range(K):
            tmp.append((pi[k] * gaussian(mu[k], sigma[k], X[i])) \
                       / sum([pi[k_] * gaussian(mu[k_], sigma[k_], X[i]) for k_ in range(K)]))
        R.append(tmp)

    return np.array(R)


def maximize(X, R, K):
    mu = []
    sigma = []
    pi = []

    N = len(X)

    for k in range(K):
        sum_rik = 0.0
        sum_rikx = np.zeros(2)
        for i in range(N):
            sum_rik += R[i][k]
            sum_rikx += R[i][k] * X[i]

        pi.append(sum_rik/N)
        mu.append(list(sum_rikx/sum_rik))

    for k in range(K):
        _sum = np.zeros((2, 2))
        for i in range(N):
            x = np.array(X[i]).reshape(2, 1)
            u = np.array(mu[k]).reshape(2, 1)
            _sum += R[i][k] * np.dot(x - u, (x - u).T)
        sigma.append(_sum / sum(R[:, k]))

    return (np.array(mu), np.array(sigma), np.array(pi))


def EM(X, K, init_theta):
    LL = 0

    theta = copy.deepcopy(init_theta)

    while True:
        LL_old = copy.deepcopy(LL)
        R = expect(X, theta)
        theta = maximize(X, R, K)
        LL = expected_complete_LL(X, R, K, theta)

        if np.abs(LL - LL_old) < 0.1:
            break

    global best_R
    global best_theta
    global best_LL
    global best_K
    best_LL.append(LL)
    best_R.append(R)
    best_theta.append(theta)
    best_K.append(K)
    #print(LL)
    return LL

best_R = []
best_theta = []
best_LL = []
best_K = []


def find_best_k(X):
    '''
    K = 3
    init_theta = get_initial_random_state(X, K)
    init_theta = kmeans(X, init_theta)
    LL = EM(X, K, init_theta)

    best_LL = LL
    best_K = K
    '''
    for K in range(2, 8):
        init_theta = get_initial_random_state(X, K)
        init_theta = kmeans(X, init_theta)
        LL = EM(X, K, init_theta)

    global best_R
    global best_theta
    global best_LL
    global best_K
    argmax = np.argmax(np.array(best_LL))
    best_LL = best_LL[argmax]
    best_theta = best_theta[argmax]
    best_R = best_R[argmax]
    best_K = best_K[argmax]

    return best_K, best_theta, best_LL, best_R


def draw_dataset(X, R):
    # Code from Jooyeon's homework
    filename = "dataset.svg"

    colors = []
    for r_i in R:
        max_r = max(r_i)
        for k in range(len(r_i)):
            if r_i[k] == max_r:
                colors.append(k + 1)
                break

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_color('#999999')
    plt.gca().spines['left'].set_color('#999999')
    plt.xlabel('x1', fontsize=20, color='#555555');
    plt.ylabel('x2', fontsize=20, color='#555555')
    plt.tick_params(axis='x', colors='#777777')
    plt.tick_params(axis='y', colors='#777777')
    plt.scatter(X[:, 0], X[:, 1], c=colors, edgecolor='none', s=30)

    plt.savefig(filename)
    # elice_utils.send_image(filename)

    plt.close()


def main():
    X = read_data("example.txt")
    best_K, best_theta, best_LL, best_R = find_best_k(X)
    draw_dataset(X, best_R)


if __name__ == '__main__':
    main()
