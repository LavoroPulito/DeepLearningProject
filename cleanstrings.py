file = open("esempioLog.txt")
tutto = file.read()
righe = tutto.split("\n")
for i in range(10):
    print(i,righe[i])
file.close()
