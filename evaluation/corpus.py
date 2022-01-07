

# class Template:
#
#     def __init__(self, template_string):
#         self.template = template_string
#         self.sentences = []
#
#     def add_sentence(self, sent):
#         self.sentences.append(sent)
# class SentencePair:
#
#     def __init__(self):

class Sentence:
    def __init__(self, sent_string):
        self.text = sent_string
        self.forms = None # a string that indicates the grammar requirements of the sentence



def fill_people_new(template_sents, people, lang, person_emotion_agreement=False) -> Dict[str,List]:
    bias2sents = defaultdict(list)

    for grammar_type in lang2grammar[lang]["people"]:
        # filter by people grammar
        mask = people_sheet["Grammar"].str.contains(grammar_type, na=False)
        filtered_people_sheet = people_sheet[mask]

        for row in filtered_people_sheet.values:
            row = [x if x != zero_out else "" for x in row] # deals with unmarked phenomenon
            person1, person2, bias_cat1, bias_cat2, bias_type = map(str.strip,row[:5])
            if person_emotion_agreement:
                person1_gender, person2_gender = map(str.strip, row[6:8]) #TODO can add validation that rows are in the right order
            # make sure neither empty
            if not person1 or not person2:
                print("one of the pairs in the following row is empty:\n {}".format(row[:5]))
                continue
            new_sent1 = re.sub(grammar2pattern[grammar_type], person1, sent)
            new_sent2 = re.sub(grammar2pattern[grammar_type], person2, sent)



            # deal with whitespace issues
            new_sent1 = re.sub("\s+", " ", new_sent1)
            new_sent2 = re.sub("\s+", " ", new_sent2)
            if (new_sent1, new_sent2) in pairs:
                continue
            #  Need to handle duplication when something is both grammar categories
            #  since in some pairs the counterfactual will be duplicated for one category
            #  TODO work out if this duplication is a problem
            pairs.add((new_sent1, new_sent2))
            cat2bias2sents[emotion_cat][bias_cat1].append(new_sent1)
            cat2bias2sents[emotion_cat][bias_cat2].append(new_sent2)

    return cat2bias2sents