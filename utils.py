from xml.etree.ElementTree import parse

def parse_review_term(path):
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