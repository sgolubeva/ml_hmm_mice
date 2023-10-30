import matplotlib.pyplot as plt 
fit_data_dict = {}
with open('fit_dict_save_BW051.txt') as fh:
    for line in fh:
        line = line.strip('\n')
        line = line.split(',')
        fit_data_dict[float(line[0])] = [float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])]

keys = list(fit_data_dict.keys())
values = list(fit_data_dict.values())

x = []
y = []

for key, val_list in zip(keys, values):
    x.extend([key] * len(val_list))
    y.extend(val_list)

plt.scatter(x, y)
plt.xlabel("State_n")
plt.ylabel("Log_likelyhood")
plt.title("Log_likelihood_BW051")
plt.savefig('Log_likelihood_BW051.png')



