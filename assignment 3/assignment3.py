import random

#k = ....
num_points = 10
#datafile to use. expected to be in upper triangular form already
dataFile = 'usnewsDistance.txt' or None

#use this with the other .txt files if you want
fileToConvert = ''

def convertFile(filein, fileout):
    with open(filein, 'r', encoding = 'utf8') as file:
        with open(fileout, 'w', encoding = 'utf8') as outfile:
            for line in file:
                temp = line.split('|')[-1]
                temp = temp.lower()
                temp = temp.replace('#','')
                temp = temp.split(' ')
                for wordindex in range(len(temp)-1,-1,-1):
                    if temp[wordindex].startswith('@') or temp[wordindex].startswith('http'):
                        temp.pop(wordindex)
                temp = ' '.join(temp)
                if not temp.endswith('\n'):
                    temp+='\n'
                outfile.write(temp)

def distance(set1 : set, set2 : set) -> float:
    return 1 - len(set1 & set2)/len(set1 | set2)

#create an upper triangular array from an input file, and store it in the output file
def computeDistance(infile, outfile):
    output = []
    array = []
    with open(infile, 'r', encoding = 'utf8') as f:
        for line in f:
            array.append(set(line.split(' ')))
    for initial in range(len(array)):
        output.append([])
        for other in range(len(array[initial+1:])):
            output[-1].append(distance(array[initial], array[other]))
    with open(outfile, 'w', encoding = 'utf8') as f:
        for line in output:
            f.write(' '.join([str(x) for x in line])+'\n')

def numLines(file):
    with open(file, encoding = 'utf8') as f:
        i = 0
        for i, _ in enumerate(f, 1):
            pass
    return i

def pointDistance(array, point1, point2) -> float:
    if point1 > point2:
        #flip the points and convert point2 to
        point1, point2 = point2, point1
        point2 = point1 - point2 - 1 # ignore the middle line
    try:
        return array[point1][point2]
    except:
        return 1.0

def pointToLine(array, point):
    total = []
    opp = 0
    for x in range(point-1, 0, -1):
        total.append(array[opp][x])
        opp += 1
    for x in array[opp]:
        total.append(x)
    return total

def SSE(cluster_base, cluster_groups, array):
    errors = []
    #TODO: iterate through cluster base and the groups together, comparing each cluster's base to the point
    for x in range(len(cluster_base)):
        error_sum = 0
        for y in cluster_groups[x]:
            error_sum += pointDistance(array, cluster_base[x], y)**2
        errors.append(error_sum)
    return errors

def closestCluster(cluster_points, matrix, point):
    #index, distance
    closest = (-1,1)
    for i, x in enumerate(cluster_points):
        dist = pointDistance(matrix, x, point)
        if dist < closest[1]:
            closest = (i,dist)
    return closest[0]

def findClusters(clust, num_clusts, array):
    dclust = [None] * num_clusts
    for x in range(len(array)):
        index = closestCluster(clust, array, x)
        if dclust[index] is None:
            dclust[index] = [x]
        else:
            dclust[index].append(x)
    return dclust

def avgPoint(arr, point):
    line = pointToLine(arr, point)
    average = sum(line)/len(arr)
    return average

def newClusters(dclust, arr):
    cs = []
    for x in range(len(dclust)):
        # index, average
        smallest = (-1,1)
        for y in dclust[x]:
            avg = avgPoint(arr, y)
            if avg < smallest[1]:
                smallest = (y, avg)
        cs.append(smallest[0])
    return cs

################################################################################

if dataFile is None:
    dataFile = 'converted'+fileToConvert
    convertFile(fileToConvert,'temp.txt')
    computeDistance('temp.txt', dataFile)

tri_array = []
count = 0
with open(dataFile) as f:
    for line in f:
        count += 1
        tri_array.append([float(x) for x in line.split(' ') if x != '\n'])

#random sample to avoid collisions
clusters = random.sample(range(count), num_points)
data_clusters = [None] * num_points
old_clusters = 0
too_high = 0

#cluster all the points per cycle
#choose new center points based on the most well connected point in the cluster to the other points

while clusters != old_clusters or too_high > 20:
    old_clusters = clusters
    # group the data around the points
    data_clusters = findClusters(clusters, num_points, tri_array)
    # center the points around the most connected data in each cluster
    clusters = newClusters(data_clusters, tri_array)

    too_high += 1
    print('Iteration num: ',too_high)
    print('Num points per cluster: ',', '.join([str(len(x)) for x in data_clusters]))
    error = [SSE(clusters, tri_array, x) for x in data_clusters]
    # print('Error per cluster: ',', '.join([str(x) for x in error]))
    print('Overall SSE: ', sum([sum(x) for x in error]))
    print('-------------------\n')

print(too_high)
print(clusters)