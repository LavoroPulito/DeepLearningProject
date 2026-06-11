file = open("logs/gen9championsvgc2026regma-2584198395.txt")
tutto = file.read()
righe = tutto.split("\n")
for i in range(10):
    print(i,righe[i])
file.close()
