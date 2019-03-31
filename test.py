from urllib.request import urlretrieve
import sys, os, time

for i in range(10000, 0, -1):
    print(i, end='\r')
    time.sleep(0.3)
    sys.stdout.flush()
