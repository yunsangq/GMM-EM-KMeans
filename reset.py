import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import copy
import numpy.linalg as linalg
# import elice_utils
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

    mu, sigma, pi = theta
    K = len(mu)

    X1 = X[:, 0]
    X2 = X[:, 1]

    while True:
        mu_old = copy.deepcopy(mu)
        labels = []
        for i in range(len(X)):
            d = []
            for j in range(K):
                d.append(np.sqrt((X1[i] - mu[j][0]) ** 2 + (X2[i] - mu[j][1]) ** 2))
            d = np.array(d)
            labels.append(np.argmin(d))

        for j in range(K):
            d_sum = 0.0
            X1_u_sum = 0.0
            X2_u_sum = 0.0
            for i in range(len(X)):
                if labels[i] == j:
                    d_sum += 1.0
                    X1_u_sum += X1[i]
                    X2_u_sum += X2[i]

            mu[j][0] = X1_u_sum / d_sum
            mu[j][1] = X2_u_sum / d_sum

        if np.allclose(mu, mu_old):
            break

    return (mu, sigma, pi)


def expected_complete_LL(X, R, K, theta):
    ll = 0
    '''
    mu, sigma, pi = theta

    for i in range(len(X)):
        for k in range(K):
            ll += np.log(pi[k] * gaussian(mu[k], sigma[k], X[i]))
    '''
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
        sum_R = sum(R[:, k])
        pi.append(sum_R / N)

        sum_RX1 = np.dot(np.array(R[:, k]), np.array(X[:, 0]))
        sum_RX2 = np.dot(np.array(R[:, k]), np.array(X[:, 1]))

        mu.append([sum_RX1 / sum_R, sum_RX2 / sum_R])
        '''
        sum_RX1M = np.sum(np.dot(np.dot(np.array(R[:, k]), np.array(X[:, 0])), np.array(X[:, 0]).T))
        sum_RX2M = np.sum(np.dot(np.dot(np.array(R[:, k]), np.array(X[:, 1])), np.array(X[:, 1]).T))

        sigma.append([(sum_RX1M / sum_R) - (np.array(mu[k]) * np.array(mu[k]).T),
                      (sum_RX2M / sum_R) - (np.array(mu[k]) * np.array(mu[k]).T)])
        '''
        

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

    return LL


def find_best_k(X):
    best_LL = None
    best_theta = None
    best_K = None
    best_R = None

    for K in range(2, 8):
        init_theta = get_initial_random_state(X, K)
        init_theta = kmeans(X, init_theta)
        LL = EM(X, K, init_theta)

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
