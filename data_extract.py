from utils import parse_review_term

data_path = './data/official_data/ABSA16_Restaurants_Train_SB1_v2.xml'
x = parse_review_term(data_path)
print(x)