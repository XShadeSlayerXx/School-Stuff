from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random

sizes = [100]#, 1000, 10000]

def getTimeDif(earlier, later):
    return (later - earlier).total_seconds()

def rand_array(amount):
    return list(np.random.permutation(amount))

def listAsc(amount):
    return list(range(amount))

def listDec(amount):
    return list(range(amount-1, -1, -1))

# def random_sort(my_list):
#     upper_bound = len(my_list)**3
#     keys = [random.randint(0,upper_bound) for x in len(my_list)]
#     temp = dict(zip(keys, my_list))
#     return [temp[x] for x in sorted(temp)]

def randomized_partition(A, p, r):
    i = random.randrange(p, r)
    A[r], A[i] = A[i], A[r]
    while p < r:
        if A[p] < A[r]:
            A[p], A[r], A[r-1] = A[r-1], A[p], A[r]
            r -= 1
        p += 1
    return i

def randomized_quicksort(A, p, r):
    if p < r:
        q = randomized_partition(A, p, r)
        randomized_quicksort(A, p, q)
        randomized_quicksort(A, q + 1, r)

for size in sizes:
    array = rand_array(size)
    # array = listDec(1000)
    copy = array[:]

    time = 0

    a = datetime.now()
    randomized_quicksort(copy, 0, len(copy)-1)
    b = datetime.now()
    time = getTimeDif(a, b)

    print(f'Size: {size} - Time: {time}')

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(0.00, time, color = 'b', width = 0.25)
    ax.set_ylabel('Time')
    m = time
    if m < 1:
        m = 1
    ax.set_yticks(np.arange(0, m, m/10))
    plt.subplots_adjust(bottom = .25, left = .25)
    plt.show()