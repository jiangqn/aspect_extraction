from xml.etree.ElementTree import parse

def parse_sentence_term(path):
    tree = parse(path)
    sentences = tree.getroot()
    data = []
    split_char = '__split__'
    for sentence in sentences:
        text = sentence.find('text')
        if text is None:
            continue
        text = text.text
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is None:
            continue
        for aspectTerm in aspectTerms:
            term = aspectTerm.get('term')
            polarity = aspectTerm.get('polarity')
            start = aspectTerm.get('from')
            end = aspectTerm.get('to')
            piece = text + split_char + term + split_char + polarity + split_char + start + split_char + end
            data.append(piece)
    return data

def parse_sentence_category(path):
    tree = parse(path)
    sentences = tree.getroot()
    data = []
    split_char = '__split__'
    for sentence in sentences:
        text = sentence.find('text')
        if text is None:
            continue
        text = text.text
        aspectCategories = sentence.find('aspectCategories')
        if aspectCategories is None:
            continue
        for aspectCategory in aspectCategories:
            category = aspectCategory.get('category')
            polarity = aspectCategory.get('polarity')
            piece = text + split_char + category + split_char + polarity
            data.append(piece)
    return data

def parse_review1_term(path):
    tree = parse(path)
    reviews = tree.getroot()
    data = []
    split_char = '__split__'
    for review in reviews:
        sentences = review.find('sentences')
        for sentence in sentences:
            text = sentence.find('text')
            if text is None:
                continue
            text = text.text
            opinions = sentence.find('Opinions')
            if opinions is None:
                continue
            for opinion in opinions:
                target = opinion.get('target')
                polarity = opinion.get('polarity')
                start = opinion.get('from')
                end = opinion.get('to')
                if (target is None) or (polarity is None) or (polarity == "") or (start is None) or (end is None):
                    continue
                piece = text + split_char + target + split_char + polarity + split_char + start + split_char + end
                data.append(piece)
    return data

def parse_review1_category(path):
    tree = parse(path)
    reviews = tree.getroot()
    data = []
    split_char = '__split__'
    for review in reviews:
        sentences = review.find('sentences')
        for sentence in sentences:
            text = sentence.find('text')
            if text is None:
                continue
            text = text.text
            opinions = sentence.find('Opinions')
            if opinions is None:
                continue
            for opinion in opinions:
                category = opinion.get('category')
                polarity = opinion.get('polarity')
                if (category is None) or (polarity is None) or (polarity == ""):
                    continue
                piece = text + split_char + category + split_char + polarity
                data.append(piece)
    return data

def parse_review2_term(path):
    tree = parse(path)
    reviews = tree.getroot()
    data = []
    split_char = '__split__'
    for review in reviews:
        sentences = review.find('sentences')
        text = ""
        for sentence in sentences:
            text += sentence.text
        opinions = review.find('Opinions')
        for opinion in opinions:
            target = opinion.get('target')
            polarity = opinion.get('polarity')
            start = opinion.get('from')
            end = opinion.get('to')
            if (target is None) or (polarity is None) or (polarity == "") or (start is None) or (end is None):
                continue
            piece = text + split_char + target + split_char + polarity + split_char + start + split_char + end
            data.append(piece)
    return data

def parse_review2_category(path):
    tree = parse(path)
    reviews = tree.getroot()
    data = []
    split_char = '__split__'
    for review in reviews:
        sentences = review.find('sentences')
        text = ""
        for sentence in sentences:
            text += sentence.find('text').text
        opinions = review.find('Opinions')
        for opinion in opinions:
            category = opinion.get('category')
            polarity = opinion.get('polarity')
            if (category is None) or (polarity is None) or (polarity == ""):
                continue
            piece = text + split_char + category + split_char + polarity
            data.append(piece)
    return data

target_path0 = '../data/official_data/processed_data/sentences_term_laptop.txt'
target_path1 = '../data/official_data/processed_data/sentences_term_restaurant.txt'

data_path0 = ['../data/official_data/SemEval-2014/Laptops_Train.xml',
                '../data/official_data/SemEval-2014/SemEval-14-ABSA-TrainData_v2 & AnnotationGuidelines/Laptop_Train_v2.xml',
                '../data/official_data/SemEval-2014/ABSA_Gold_TestData/Laptops_Test_Gold.xml',
                '../data/official_data/SemEval-2015/ABSA15_LaptopsTrain/ABSA-15_Laptops_Train_Data.xml',
              '../data/official_data/SemEval-2015/ABSA15_Laptops_Test.xml']
data_path1 = ['../data/official_data/SemEval-2014/Restaurants_Train.xml',
                '../data/official_data/SemEval-2014/SemEval-14-ABSA-TrainData_v2 & AnnotationGuidelines/Restaurants_Train_v2.xml',
                '../data/official_data/SemEval-2014/ABSA_Gold_TestData/Restaurants_Test_Gold.xml',
                '../data/official_data/SemEval-2015/ABSA15_RestaurantsTrain/ABSA-15_Restaurants_Train_Final.xml',
              '../data/official_data/SemEval-2015/ABSA15_Restaurants_Test.xml']

target_path = [target_path0, target_path1]
data_path = [data_path0, data_path1]
parsers = [parse_sentence_term, parse_sentence_term]

for i in range(2):
    file = open(target_path[i], 'w', encoding=u'utf-8')
    data = set()
    for path in data_path[i]:
        data |= set(parsers[i](path))
    for piece in data:
        file.write(piece.lower() + '\n')