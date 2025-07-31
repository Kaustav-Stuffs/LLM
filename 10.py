def print_pattern(n):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if (i == 1 and j % 2 != 0) or (i == n and j % 2 != 0):
                print("*", end="")
            elif i == j or j == (n - i + 1):
                print("*", end="")
            else:
                print(" ", end="")
        print()

print_pattern(11)
