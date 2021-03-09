# from allennlp.predictors import Predictor
from logging import exception
import hanlp
import json

class SRLPredictor(object):
    def __init__(self,SRL_MODEL_PATH):
        # use the model from allennlp for simlicity.
        # self.predictor = Predictor.from_path(SRL_MODEL_PATH)
        # self.predictor._model = self.predictor._model.cuda()  # this can only support GPU computation
        pass

    def predict(self, sent):
        # return self.predictor.predict(sentence=sent)
        pass


def get_tags(srl_predictor, tok_text, tag_vocab):
    if srl_predictor == None:
        try:
            srl_result = json.loads(tok_text)  # can load a pre-tagger dataset for quick evaluation
        except:
            print("DATA error!!{}".format(tok_text))
            return None
    else:
        srl_result = srl_predictor.predict(tok_text)
    sen_verbs = srl_result['verbs']
    sen_words = srl_result['words']

    sent_tags = []
    if len(sen_verbs) == 0:
        sent_tags = [["O"] * len(sen_words)]
    else:
        for ix, verb_tag in enumerate(sen_verbs):
            sent_tag = sen_verbs[ix]['tags']
            for tag in sent_tag:
                if tag not in tag_vocab:
                    tag_vocab.append(tag)
            sent_tags.append(sent_tag)

    return sen_words, sent_tags


class HanlpSRLPredictor(object):

    def __init__(self):
        self.model = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH) # 世界最大中文语料库

    def predict(self, query):
        res = self.model([query])
        tokens = res['tok/fine'][0]
        start = 0
        index_map = []
        for token in tokens:
            index_map.append((start, len(token) + start))
            start = len(token) + start
        all_srl_list = []
        for srl in res['srl'][0]:
            one_srl_dict = {}
            srl_list = ["o"] * len(query)
            for srl_tuple in srl:
                token = srl_tuple[0]
                tag = srl_tuple[1]
                start_index = srl_tuple[2]
                end_index = srl_tuple[3]
                char_start = index_map[start_index][0]
                char_end = index_map[end_index - 1][1]
                if tag == "PRED":
                    one_srl_dict["verb"] = tokens[start_index]
                srl_list[char_start] = "B-" + tag
                if char_end - char_start > 1:
                    srl_list[char_start+1: char_end] = ["I-" + tag]*(char_end - char_start - 1)
            one_srl_dict["tags"] = srl_list
            all_srl_list.append(one_srl_dict)
        res_dict = {
            "verbs": all_srl_list,
            "words": [w for w in query]
        }
        return res_dict
    
