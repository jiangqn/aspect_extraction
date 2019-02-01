data_path = '../data/restaurant/train.txt'
base_path = '../data/restaurant/reviews/'

reviews = open(data_path, 'r', encoding=u'utf-8').readlines()
num = len(reviews)
file = None

for i in range(num):
    if i % 100 == 0:
        file_name = '%s%04d.txt' % (base_path, i // 100)
        file = open(file_name, 'w', encoding=u'utf-8')
    file.write(reviews[i])