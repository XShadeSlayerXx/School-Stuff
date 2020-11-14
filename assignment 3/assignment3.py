import random

#k = ....
min_points = 3
max_points = 18
num_point_increment = 3

#datafile to use. expected to be in upper triangular form already
dataFile = None# or 'usnewsDistance.txt'

#use this with the other .txt files if you want
# adds a converted file called 'converted'+filename -> e.g. 'usnews.txt' -> 'convertedusnews.txt'
# this can be used directly in the datafile^^^ variable for subsequent trials
fileToConvert = 'latimeshealth.txt'

acceptable_error = .0001

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
        #flip the points and convert point2 to left hand side coordinates
        point1, point2 = point2, point1
    point2 = point2 - point1 - 1 # ignore the middle line
    try:
        return array[point1][point2]
    except:
        raise(Exception)

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
tri_array = tri_array[:-1]

#random sample to avoid collisions
cluster_seeds = random.sample(range(count), max_points)

for num_points in range(min_points, max_points, num_point_increment):
    clusters = cluster_seeds.copy()[:num_points]
    data_clusters = [None] * num_points
    old_sse = float('inf')
    older_sse = 0
    new_sse = float('-inf')
    too_high = 0

    print('-------------------\n')
    print('Number of clusters: ',num_points)
    print('Cluster Start Locations: ', ', '.join([str(x) for x in clusters]))

    #cluster all the points per cycle
    #choose new center points based on the most well connected point in the cluster to the other points

    while old_sse != new_sse and too_high < 20 and older_sse != new_sse:
        older_sse = old_sse
        old_sse = new_sse
        # group the data around the points
        data_clusters = findClusters(clusters, num_points, tri_array)
        # center the points around the most connected data in each cluster
        clusters = newClusters(data_clusters, tri_array)

        too_high += 1
    print('Cluster End Locations:   ',', '.join([str(x) for x in clusters]))
    print('Number of points per cluster: ',', '.join([str(len(x)) for x in data_clusters]))
    error = SSE(clusters, data_clusters, tri_array)
    print('Error per cluster: ',', '.join([str(x) for x in error]))
    new_sse = sum(error)
    print('Overall SSE: ', new_sse)
