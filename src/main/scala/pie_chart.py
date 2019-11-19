import matplotlib.pyplot as plt
import os

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
dir = "mostPopularWords_LogReg"
files = os.listdir(dir)
file = [files[i] for i in range(len(files)) if files[i][-4:-1]+files[i][-1]=='0000'][0]

lines = []
with open(dir + "/" + file) as f:
   line = f.readline()
   while line:
       lines.append(line)
       line = f.readline()

labels = []
sizes = []
for line in lines:
    parts = line.strip('(').split(')')[0].split(',')
    print(parts)
    labels.append(parts[1])
    sizes.append(int(parts[0]))


fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.savefig('pie_chart.png')
