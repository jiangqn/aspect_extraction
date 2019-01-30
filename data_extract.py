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
            opinions = sentence.find('opinions')
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