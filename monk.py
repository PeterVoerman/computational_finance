with open("monk.txt", "w+") as f:
    for i in range(10000):
        for j in range(100):
            f.write("{\huge \emoji{gorilla}}")
        f.write('\n')