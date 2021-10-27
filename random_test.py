import random

lst = [random.randint(1, 20) for i in range(10)]
ls0 = []
ls1 = []
print(lst)
for i in range(10):
    count = 1
    if lst[i] in ls0:
        continue
    for j in range(i + 1, len(lst)):
        if lst[i] == lst[j]:
            count += 1
        if count >= 2:
            ls0.append(lst[i])
        else:
            ls1.append(lst[i])
    print(lst[i], count)

