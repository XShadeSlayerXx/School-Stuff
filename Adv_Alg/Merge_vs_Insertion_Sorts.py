from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

sizes = [100, 1000, 10000, 100000]
types = [0,1,2]

def listRand(amount):
    return list(np.random.permutation(amount))

def listAsc(amount):
    return list(range(amount))

def listDec(amount):
    return list(range(amount-1, -1, -1))

def getCase(which, amount):
    if which == 0:
        return listAsc(amount)
    elif which == 1:
        return listDec(amount)
    else:
        return listRand(amount)

def getTimeDif(earlier, later):
    return (later - earlier).total_seconds()

def merge(A, p, q, r):
    L = A[p:q]
    R = A[q:r]
    L.append(float('inf'))
    R.append(float('inf'))
    i, j = 0, 0
    for k in range(p, r):
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1

def mergeSort(A, p, r):
    if r - p > 1:
        q = (p + r)//2
        mergeSort(A, p, q)
        mergeSort(A, q, r)
        merge(A, p, q, r)

def insertSort(A):
    for i in range(1, len(A)):
        currentVal = A[i]
        currentPos = i

        while currentPos > 0 and A[currentPos - 1] > currentVal:
            A[currentPos] = A[currentPos-1]
            currentPos -= 1

        A[currentPos] = currentVal

for size in sizes:
    mergeTimes = []
    insertTimes = []
    for type in types:
        array = getCase(type, size)
        copy = array[:]

        a = datetime.now()
        mergeSort(copy, 0, len(array))
        b = datetime.now()
        mergeTimes.append(getTimeDif(a, b))

        a = datetime.now()
        insertSort(array)
        b = datetime.now()
        insertTimes.append(getTimeDif(a, b))

    print('\n'.join([f'Case {i}: Merge - {mergeTimes[i]} vs Insert - {insertTimes[i]}' for i in range(len(mergeTimes))]))

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    X = np.arange(len(mergeTimes))
    ax.bar(X + 0.00, mergeTimes, color = 'b', width = 0.25)
    ax.bar(X + 0.25, insertTimes, color = 'r', width = 0.25)
    ax.set_ylabel('Time')
    m = max(mergeTimes[-1], insertTimes[-1])
    if m < 1:
        m = 1
    ax.set_yticks(np.arange(0, m, m/10))
    ax.legend(labels = ['Merge', 'Insert'])
    plt.subplots_adjust(bottom = .25, left = .25)
    plt.show()