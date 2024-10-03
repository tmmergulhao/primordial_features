import numpy as np
test1 = np.loadtxt('unit_test_logprobs.txt')
test2 = np.loadtxt('unit_test_old_ver_logprobs.txt')
print(np.all(test1/test2 == 1))