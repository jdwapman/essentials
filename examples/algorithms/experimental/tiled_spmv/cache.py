# Simple script to annotate memory accesses as hits or L1/L2 misses
import sys
import numpy as np

# The first argument has the filename
filename = sys.argv[1]

# Open the file
step1 = []
step2 = []
step3 = []
with open(filename, 'r') as f:
  # Process step 1
  # Ignore the first line of the file
  lines = f.readlines()

  s1 = 0
  s2 = 0
  s3 = 0
  for idx, line in enumerate(lines):
    # Check if line contains the string "STEP2"
    if "STEP2" in line:
      s2 = idx

    if "STEP3" in line:
      s3 = idx


print(s1, s2, s3)

step1 = lines[s1+1:s2]
step2 = lines[s2+1:s3]
step3 = lines[s3+1:]

for idx, s in enumerate(step1):
  # s is in the format "idx,val"
  # Want to split and convert to numbers
  s = s.split(",")
  s[0] = int(s[0])
  s[1] = float(s[1])
  step1[idx] = s

# Do the same thing for step 2 and step 3
for idx, s in enumerate(step2):
  s = s.split(",")
  s[0] = int(s[0])
  s[1] = float(s[1])
  step2[idx] = s

for idx, s in enumerate(step3):
  s = s.split(",")
  s[0] = int(s[0])
  s[1] = float(s[1])
  step3[idx] = s

# Convert to numpy
step1 = np.array(step1)
step2 = np.array(step2)
step3 = np.array(step3)

# Plot step1
import matplotlib.pyplot as plt
# Use the first column as the x-axis
plt.plot(step1[:,0], step1[:,1])
# Save the figure
plt.savefig("step1.png")

# Do again for step2 and step3
plt.plot(step2[:,0], step2[:,1])
plt.savefig("step2.png")

plt.plot(step3[:,0], step3[:,1])
plt.savefig("step3.png")

# Get a histogram of the values of step1
# Get a new figure
plt.figure()
h = np.histogram(step1[:,1], bins=1000)
plt.plot(h[1][:-1], h[0])
plt.xlabel('Access Time (sec)')
plt.ylabel('Frequency')
plt.title('Histogram of Access Times (Step 1)')
plt.savefig("step1_hist.png")


# Step 2 and 3
plt.figure()
h = np.histogram(step2[:,1], bins=1000)
plt.plot(h[1][:-1], h[0])
plt.xlabel('Access Time (sec)')
plt.ylabel('Frequency')
plt.title('Histogram of Access Times (Step 2)')
plt.savefig("step2_hist.png")

plt.figure()
h = np.histogram(step3[:,1], bins=1000)
plt.plot(h[1][:-1], h[0])
plt.xlabel('Access Time (sec)')
plt.ylabel('Frequency')
plt.title('Histogram of Access Times (Step 3)')
plt.savefig("step3_hist.png")
