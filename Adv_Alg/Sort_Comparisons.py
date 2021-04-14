from datetime import datetime
import numpy as np

# 0 is counting, 1 is radix, 2 is bucket
types = [0,1,2]

# 0 is ascending, 1 is descending, 2 is random
styles = [0,1,2]

# the size of the array
size = 10000

# to ensure the random arrays are equivalent between the sorts
random_array = list(np.random.permutation(size))

# set to True for array before/after
debug_output = False

def getTimeDif(earlier, later):
    return (later - earlier).total_seconds()

# def rand_array(amount):
#     return list(np.random.permutation(amount))

def listAsc(amount):
    return list(range(amount))

def listDec(amount):
    return list(range(amount-1, -1, -1))

def get_array_order(which, size):
    global random_array
    if which == 0: #asc
        return 'Ascending', listAsc(size)
    elif which == 1: #dec
        return 'Descending', listDec(size)
    else: #rand
        return 'Random', random_array

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def counting_sort(A):
    C = [0]*len(A)
    B = [0]*len(A)
    for x in A:
        C[x] += 1
    for x in range(1, len(C)):
        C[x] += C[x-1]
    for x in range(len(A) - 1, -1, -1):
        B[C[A[x]]-1] = A[x]
        C[A[x]] -= 1
    return B

def radix_counting_sort(A, digit):
    C = [0] * len(A)
    B = [0] * 10
    for i in A:
        index = (i // digit)
        B[index%10] += 1
    for i in range(1, len(B)):
        B[i] += B[i-1]
    for i in range(len(A)-1, -1, -1):
        index = A[i] // digit
        C[B[index%10]-1] = A[i]
        B[index%10] -= 1
    return C

def radix_sort(A):
    high = max(A)
    low = 1
    while low < high:
        A = radix_counting_sort(A, low)
        low *= 10
    return A

def bucket_sort(A):
    B = [list() for _ in range(len(A))]
    for i in range(len(A)):
        B[A[i]].append(A[i])
    for i in B:
        insertion_sort(i)
    place = 0
    for x in B:
        for y in x:
            A[place] = y
            place += 1
    return A

def sort_type(which, my_array):
    if which == 0:
        return 'Counting Sort', counting_sort(my_array)
    elif which == 1:
        return 'Radix Sort', radix_sort(my_array)
    else:
        return 'Bucket Sort', bucket_sort(my_array)

for sort in types:
    for order in styles:
        word, array = get_array_order(order, size)
        copy = array[:]

        time = 0

        if debug_output: print(copy)
        a = datetime.now()
        sort_name, sorted_array = sort_type(sort, copy)
        b = datetime.now()
        if debug_output: print(sorted_array)

        time = getTimeDif(a, b)

        print(f'Sort Type: {sort_name} - Order: {word} - Time: {time}')
        if debug_output: print('-'*30)
    print('='*30)