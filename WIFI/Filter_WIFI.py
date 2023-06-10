import pandas as pd

# read the file
with open('EastWestAll(06.09_20.00_added)/east.csv', 'r') as f:
    data = f.readlines()

# create a list to hold all SSIDs and another to hold the filtered lines
ssids = []
filtered_lines = []

# list of desired SSIDs
desired_ssids = ["TUD-facility", "tudelft-dastud", "eduroam", "SmartLife-5DFF"]

# extract ssid from each line and add it to the list
for line in data:
    line = line.strip()  # remove newline character at the end of line
    ssid = line.split(', ')[0].split(': ')[1]  # get the SSID directly
    ssids.append(ssid)  # add it to the list
    if ssid in desired_ssids:
        filtered_lines.append(line)  # add the whole line to the filtered list if the SSID is desired

# convert the list into a pandas Series and count the values
ssid_counts = pd.Series(ssids).value_counts()

# print the counts
print(ssid_counts)

# write the filtered lines to a new CSV file
with open('EastWestAll(06.09_20.00_added)/east_filtered.csv', 'w') as f:
    for line in filtered_lines:
        f.write(line + '\n')
        
        
        
        
        
        
# read the file
with open('EastWestAll(06.09_20.00_added)/west.csv', 'r') as f:
    data = f.readlines()

# create a list to hold all SSIDs and another to hold the filtered lines
ssids = []
filtered_lines = []

# list of desired SSIDs
desired_ssids = ["TUD-facility", "tudelft-dastud", "eduroam", "SmartLife-5DFF"]

# extract ssid from each line and add it to the list
for line in data:
    line = line.strip()  # remove newline character at the end of line
    ssid = line.split(', ')[0].split(': ')[1]  # get the SSID directly
    ssids.append(ssid)  # add it to the list
    if ssid in desired_ssids:
        filtered_lines.append(line)  # add the whole line to the filtered list if the SSID is desired

# convert the list into a pandas Series and count the values
ssid_counts = pd.Series(ssids).value_counts()

# print the counts
print(ssid_counts)

# write the filtered lines to a new CSV file
with open('EastWestAll(06.09_20.00_added)/west_filtered.csv', 'w') as f:
    for line in filtered_lines:
        f.write(line + '\n')








# read the file
with open('Floor(06.09_20.00_added)/floor1.csv', 'r') as f:
    data = f.readlines()

# create a list to hold all SSIDs and another to hold the filtered lines
ssids = []
filtered_lines = []

# list of desired SSIDs
desired_ssids = ["TUD-facility", "tudelft-dastud", "eduroam"]

# extract ssid from each line and add it to the list
for line in data:
    line = line.strip()  # remove newline character at the end of line
    ssid = line.split(', ')[0].split(': ')[1]  # get the SSID directly
    ssids.append(ssid)  # add it to the list
    if ssid in desired_ssids:
        filtered_lines.append(line)  # add the whole line to the filtered list if the SSID is desired

# convert the list into a pandas Series and count the values
ssid_counts = pd.Series(ssids).value_counts()

# print the counts
print(ssid_counts)

# write the filtered lines to a new CSV file
with open('Floor(06.09_20.00_added)/floor1_filtered.csv', 'w') as f:
    for line in filtered_lines:
        f.write(line + '\n')
        
        
        
        
        
        
        
        
        
        
        
        
        
# read the file
with open('Floor(06.09_20.00_added)/floor2.csv', 'r') as f:
    data = f.readlines()

# create a list to hold all SSIDs and another to hold the filtered lines
ssids = []
filtered_lines = []

# list of desired SSIDs
desired_ssids = ["TUD-facility", "tudelft-dastud", "eduroam"]

# extract ssid from each line and add it to the list
for line in data:
    line = line.strip()  # remove newline character at the end of line
    ssid = line.split(', ')[0].split(': ')[1]  # get the SSID directly
    ssids.append(ssid)  # add it to the list
    if ssid in desired_ssids:
        filtered_lines.append(line)  # add the whole line to the filtered list if the SSID is desired

# convert the list into a pandas Series and count the values
ssid_counts = pd.Series(ssids).value_counts()

# print the counts
print(ssid_counts)

# write the filtered lines to a new CSV file
with open('Floor(06.09_20.00_added)/floor2_filtered.csv', 'w') as f:
    for line in filtered_lines:
        f.write(line + '\n')
        
        











# read the file
with open('Floor(06.09_20.00_added)/floor3.csv', 'r') as f:
    data = f.readlines()

# create a list to hold all SSIDs and another to hold the filtered lines
ssids = []
filtered_lines = []

# list of desired SSIDs
desired_ssids = ["TUD-facility", "tudelft-dastud", "eduroam"]

# extract ssid from each line and add it to the list
for line in data:
    line = line.strip()  # remove newline character at the end of line
    ssid = line.split(', ')[0].split(': ')[1]  # get the SSID directly
    ssids.append(ssid)  # add it to the list
    if ssid in desired_ssids:
        filtered_lines.append(line)  # add the whole line to the filtered list if the SSID is desired

# convert the list into a pandas Series and count the values
ssid_counts = pd.Series(ssids).value_counts()

# print the counts
print(ssid_counts)

# write the filtered lines to a new CSV file
with open('Floor(06.09_20.00_added)/floor3_filtered.csv', 'w') as f:
    for line in filtered_lines:
        f.write(line + '\n')