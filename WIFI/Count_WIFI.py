import pandas as pd

# read the file
with open('Floor(06.09_20.00_added)/floor3.csv', 'r') as f:
    data = f.readlines()

# create a list to hold all SSIDs
ssids = []

# extract ssid from each line and add it to the list
for line in data:
    line = line.strip()  # remove newline character at the end of line
    ssid = line.split(', ')[0].split(': ')[1]  # get the SSID directly
    ssids.append(ssid)  # add it to the list

# convert the list into a pandas Series and count the values
ssid_counts = pd.Series(ssids).value_counts()

# print the counts
print(ssid_counts)
