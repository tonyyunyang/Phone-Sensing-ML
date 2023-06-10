import pandas as pd

# read the file
name = 'EastWestAll(06.09_20.00_added)'


full = name + '/' + 'east' + '.csv'
with open(full, 'r') as f:
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
save = name + '/' + 'east_filtered.csv'
with open(save, 'w') as f:
    for line in filtered_lines:
        f.write(line + '\n')
        
        
        
        
        
        
        
        
        
        
# read the file
full = name + '/' + 'west' + '.csv'
with open(full, 'r') as f:
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
save = name + '/' + 'west_filtered.csv'
with open(save, 'w') as f:
    for line in filtered_lines:
        f.write(line + '\n')












name2 = 'Floor(06.09_20.00_added)'

full = name2 + '/' + 'floor1' + '.csv'
# read the file
with open(full, 'r') as f:
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
save = name2 + '/' + 'floor1_filtered.csv'
with open(save, 'w') as f:
    for line in filtered_lines:
        f.write(line + '\n')
        
        
        
        
        
        
        
        
        
        
        
        
        
# read the file
full = name2 + '/' + 'floor2' + '.csv'
with open(full, 'r') as f:
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
save = name2 + '/' + 'floor2_filtered.csv'
with open(save, 'w') as f:
    for line in filtered_lines:
        f.write(line + '\n')
        
        











# read the file
full = name2 + '/' + 'floor3' + '.csv'
with open(full, 'r') as f:
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
save = name2 + '/' + 'floor3_filtered.csv'
with open(save, 'w') as f:
    for line in filtered_lines:
        f.write(line + '\n')