with open("rl_random_test.txt", "r") as f:
    lines = f.readlines()

with open("rl_random_test_.txt", "w") as f:
    for i, line in enumerate(lines):
        name, time, reward, error = line.strip().split(' ')
        time = float(time)
        reward = float(reward)
        error = float(error) / 255**2
        string = "{} & {:.2f} & {:.2f} & {:.4f} \\\\ \\hline\n".format(i+1, time, reward, error)
        f.write(string)