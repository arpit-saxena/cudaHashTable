import random

num_instructs = 20000
instructs = ["INSERT", "DELETE", "FIND"]
minInt = 0
maxInt = 2 ** 20

print(1)
print(num_instructs)
for i in range(num_instructs):
    print(f"{random.choice(instructs)} {random.randint(minInt, maxInt)}")