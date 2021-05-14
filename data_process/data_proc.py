
import sys, os, time
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))# 当前文件的上层目录
sys.path.append(BASE_DIR)

import json
from os import sep
from tools.common import load_data2df, get_time_stamp

import hanlp
import pandas as pd
from pandas.core.indexes.base import ensure_index
from pandas.core.indexes.datetimes import bdate_range

class HanlpSRLPredictor(object):

    def __init__(self):
        self.model = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH) # 世界最大中文语料库

    def predict(self, query):
        res_dict = {
            "verbs": [],
            "words": [w for w in query]
        } 
        try:
            res = self.model([query])
        except:
            return res_dict

        tokens = res['tok/fine'][0]
        start = 0
        index_map = []
        for token in tokens:
            index_map.append((start, len(token) + start))
            start = len(token) + start
        all_srl_list = []
        for srl in res['srl'][0]:
            one_srl_dict = {}
            srl_list = ["O"] * len(query)
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
            one_srl_dict["description"] = srl
            one_srl_dict["tags"] = srl_list
            all_srl_list.append(one_srl_dict)
        res_dict = {
            "verbs": all_srl_list,
            "words": [w for w in query]
        }
        return res_dict


hanlpSRLPredictor = HanlpSRLPredictor()
def data_format(input_file, output_file):
    '''
    input_file: query\ttag
    output_file: id\tquery\tsrl\ttag
    '''
    out_pd = pd.DataFrame()
    hanlpSRLPredictor = HanlpSRLPredictor()
    sample_id = 0
    with open(input_file, 'r') as f:
        lines = f.readlines()

    fw = open(output_file, 'w') 
    count = 0
    for line in lines:
        line_arr = line.strip('\n').split("\t")
        query = line_arr[0]
        label = line_arr[1]
        srl= hanlpSRLPredictor.predict(query)
        count += 1
        # new_row = {'id': [sample_id], 'query':[query], 'srl':[json.dumps(srl)], 'label': [label]}
        # row_pd = pd.DataFrame(new_row)
        # out_pd = out_pd.append(row_pd)
        sample_id += 1
        out_str = "{}\t{}\t{}\t{}\n".format(sample_id, query, json.dumps(srl, ensure_ascii=False), label)
        fw.write(out_str)
    # out_pd.to_csv(output_file, sep='\t', index=False)
    fw.close()
    return count

def query_format(query, tag=None):
    if not tag:
        import HanlpSRLPredictor
        hanlpSRLPredictor = HanlpSRLPredictor()
        tag = hanlpSRLPredictor.predict(query)
    label = "1"
    sample_id = 0
    out_str = "{}\t{}\t{}\t{}\n".format(sample_id, query, json.dumps(tag, ensure_ascii=False), label)
    return out_str


def srl_data_analyze(srl_data_file):
    srl_data = load_data2df(srl_data_file)
    srl_data.columns = ['id', 'query', 'srl', 'label']

    srl_json_data = srl_data['srl'].apply(parse_json)
    srl_count_data = srl_json_data.apply(srl_count)
    total_samples = srl_count_data.size
    srl_counts = srl_count_data.sum()
    srl_aspects_count = srl_json_data.apply(srl_aspect_count)
    srl_aspects_static = srl_aspects_count.value_counts()
    srl_avg_aspect, srl_max_aspect = srl_aspects_count.sum()/srl_counts, srl_json_data.apply(srl_aspect_count).max()
    print(f'total:{total_samples}, srl_counts:{srl_counts}, srl_avg_aspect:{srl_avg_aspect}, srl_max_aspect:{srl_max_aspect}')
    # 统计aspect个数信息
    print(f'srl_apects_count:\n{srl_aspects_static}')
    # 统计tag分布
    srl_json_data.apply(srl_tag_count)
    srl_tag_static = pd.value_counts(tag_all)
    print(f'srl_tags_aspect:{srl_tag_static}')




tag_all = []
def srl_tag_count(row):
    if len(row['verbs']) == 0:
        pass
    else:
        for one_aspect in row['verbs']:
            tag_all.extend(one_aspect['tags'])

def srl_count(row):
    if len(row['verbs']) == 0:
        return 0
    else:
        return 1

def srl_aspect_count(row):
    return len(row['verbs'])

def parse_json(row):
    json_object = json.loads(row)    
    return json_object


def get_srl_label():
    input_file = './data/intention2_v2/test.txt'
    output_file = './data/intention2_v2/test.tsv_tag'
    data_format(input_file, output_file)
    input_file = './data/intention2_v2/dev.txt'
    output_file = './data/intention2_v2/dev.tsv_tag'
    data_format(input_file, output_file)
    input_file = './data/intention2_v2/train.txt'
    output_file = './data/intention2_v2/train.tsv_tag'
    data_format(input_file, output_file)

def get_srl_label_multi_data(data_dir, task_name=None):
    if task_name:
        task_name += '_'
    else:
        task_name = ''
    count = 0
    input_file = data_dir + '/{}test.txt'.format(task_name)
    output_file = data_dir + '/test.tsv_tag'
    count += data_format(input_file, output_file)
    input_file = data_dir + '/{}dev.txt'.format(task_name)
    output_file = data_dir + '/dev.tsv_tag'
    count += data_format(input_file, output_file)
    input_file = data_dir + '/{}train.txt'.format(task_name)
    output_file = data_dir + '/train.tsv_tag'
    count += data_format(input_file, output_file)
    return count 

if __name__ == '__main__':
    # srl_file = './data/intention2_v2/test.tsv_tag'
    # srl_file_multi_data_dir = './data/Intention170_binary'
    # task_name = "General_Intention_170_V1_binary"
    # time_start = get_time_stamp()
    # count = get_srl_label_multi_data(srl_file_multi_data_dir, task_name)
    # time_end = get_time_stamp()
    # time_cost = (time_end - time_start).seconds
    # time_cost_per = 1.0*time_cost/count
    # print("time_cost_total:{}, time_cost_one:{}".format(time_cost, time_cost_per))

    input_file = './data/intention2_v2/no_label_data.txt'
    output_file = './data/intention2_v2/no_label_data.txt.tsv_tag'
    data_format(input_file, output_file)
    
    # srl_data_analyze(srl_file)
