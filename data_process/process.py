data_path = '../data/official_data/processed_data/sentences_term_restaurant.txt'

file = open(data_path)
data_dict = {}

for line in file.readlines():
    line = line.rstrip().split('__split__')
    if line[0] in data_dict:
        data_dict[line[0]].append([int(line[3]), int(line[4])])
    else:
        data_dict[line[0]] = [[int(line[3]), int(line[4])]]