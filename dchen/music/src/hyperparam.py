import numpy as np

C1 = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
C2 = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 3, 10, 30, 100, 300, 1000]
C3 = [0.01, 0.1, 0.3, 1, 2, 4, 6, 8, 10, 20, 30, 100, 300, 1000]
P = [1, 2, 3, 4, 5, 6]

nsamples = 30

for n in range(nsamples):
    c1 = np.random.choice(C1, 1)
    c2 = np.random.choice(C2, 1)
    c3 = np.random.choice(C3, 1)
    p = np.random.choice(P, 1)
    print('%g %g %g %g' % (c1, c2, c3, p))

