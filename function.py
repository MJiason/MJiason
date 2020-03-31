from scipy.stats import t, f


def table_student(prob, n, m):
    x_vec = [i * 0.0001 for i in range(int(5 / 0.0001))]
    par = 0.5 + prob / 0.1 * 0.05
    f3 = (m - 1) * n
    for i in x_vec:
        if abs(t.cdf(i, f3) - par) < 0.000005:
            return i


def table_fisher(prob, n, m, d):
    x_vec = [i * 0.001 for i in range(int(10 / 0.001))]
    f3 = (m - 1) * n
    for i in x_vec:
        if abs(f.cdf(i, n - d, f3) - prob) < 0.0001:
            return i
