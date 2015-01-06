import ipdb
import random
import sys

def reservoir_sampler(file, reservior_size, max_iter=-1, delimiter='\t'):
    reservoir = []
    cnt = 0
    #ipdb.set_trace()
    for line in file:
        line = stripped(line)
        print reservior_size, cnt
        if cnt < reservior_size:
            reservoir.append(line.rstrip('\n').split(delimiter))
            cnt += 1
            continue
        pick = random.randint(0, cnt)
        if pick <= reservior_size-1:
            reservoir[pick] = line.rstrip('\n').split(delimiter)
        cnt += 1

        if max_iter != -1 and cnt > max_iter:
            return reservoir
    return reservoir


# strips control characters
def stripped(x):
    return "".join([i for i in x if 31 < ord(i) < 127])
 


print "Opening up", sys.argv[1]
file = open(sys.argv[1])

print "Calling reservoir sampler"
data = reservoir_sampler(file, int(sys.argv[2]))

print "Writing to csv"
import csv
with open("output.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(data)

