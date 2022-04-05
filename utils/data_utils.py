import csv
import json

us_reviews_cat = ['Wireless_v1_00', 'Watches_v1_00', 'Video_Games_v1_00', 'Video_DVD_v1_00', 'Video_v1_00', 'Toys_v1_00',
                  'Tools_v1_00', 'Sports_v1_00', 'Software_v1_00', 'Shoes_v1_00', 'Pet_Products_v1_00',
                  'Personal_Care_Appliances_v1_00', 'PC_v1_00', 'Outdoors_v1_00', 'Office_Products_v1_00',
                  'Musical_Instruments_v1_00', 'Music_v1_00', 'Mobile_Electronics_v1_00', 'Mobile_Apps_v1_00',
                  'Major_Appliances_v1_00', 'Luggage_v1_00', 'Lawn_and_Garden_v1_00', 'Kitchen_v1_00', 'Jewelry_v1_00',
                  'Home_Improvement_v1_00', 'Home_Entertainment_v1_00', 'Home_v1_00', 'Health_Personal_Care_v1_00',
                  'Grocery_v1_00', 'Gift_Card_v1_00', 'Furniture_v1_00', 'Electronics_v1_00', 'Digital_Video_Games_v1_00',
                  'Digital_Video_Download_v1_00', 'Digital_Software_v1_00', 'Digital_Music_Purchase_v1_00',
                  'Digital_Ebook_Purchase_v1_00', 'Camera_v1_00', 'Books_v1_00', 'Beauty_v1_00', 'Baby_v1_00',
                  'Automotive_v1_00', 'Apparel_v1_00', 'Digital_Ebook_Purchase_v1_01', 'Books_v1_01', 'Books_v1_02']
marc_reviews_colummns = ["stars", "product_category", "product_id", "review_title", "language", "review_body", "review_id", "reviewer_id"]
us_reviews_columns = ["marketplace", "customer_id", "review_id", "product_id", "product_parent", "product_title",
                      "product_category", "star_rating", "helpful_votes", "total_votes", "vine", "verified_purchase",
                      "review_headline", "review_body", "review_date"]
CSV_HEADERS = ["us_product_cat", "marc_product_cat", "uncertain?"]

# write to csv
BY_SEED_HEADERS = ["bias_cat_1_mean", "bias_cat_2_mean", "performance_gap", "statistical_significance",
           "bias_cat_1_name", "bias_cat_2_name", "bias_type", "steps", "lang", "model_name"]
OVERALL_HEADERS = BY_SEED_HEADERS[:-1]
EMO_ADDITION = ["emotion"]
CSV_BASE = "{}_{}" # lang_seed or lang_overall

# write out data for boxplots or violinplots, with all labels
LABEL_HEADERS = ["label_1", "label_2", "performance_gap", "bias_cat_1", "bias_cat_2", "bias_type", "steps", "lang"]
EVAL_CORPUS_PATH = "/home/s1948359/multilingual_sentiment_analysis/evaluation/corpora/EEC_{}.json"


def load_corpus(lang):
    with open(EVAL_CORPUS_PATH.format(lang), "r") as fin:
        test_corpus = json.load(fin)
    return test_corpus


def load_cat_mapping(filepath: str) -> dict:
    cat2other_cat = {}
    with open(filepath, "r", newline="") as fin:
        file = csv.DictReader(fin)
    for row in file:
        cat2other_cat[row[CSV_HEADERS[0]]] = row[CSV_HEADERS[1]]

    return cat2other_cat


def get_label_from_emotion(emotion_cat):
    if emotion_cat == "no_emotion":
        return "neutral"
    if emotion_cat in {"anger", "sadness", "fear"}:
        return "negative"
    if emotion_cat == "joy":
        return "positive"

def convert_to_polarity(label):
    if label > 3:
        return "positive"
    elif label < 3:
        return "negative"
    else:
        return "neutral"

