# coding=utf-8
# 以阅读理解的语料为数据集，训练摘要模型
import os, json
import xlrd  # 引入模块
import random
import sys

sys.path.append("..")
from compute_lcs import _compute_lcs
from curLine_file import curLine


def process(corpus_folder, save_folder, type_list):
    length_abstract_list = []
    length_article_list = []
    for corpus_type in type_list:
        raw_file_name = "%s_ch.json" % corpus_type
        raw_file = os.path.join(corpus_folder, raw_file_name)
        with open(raw_file, "r") as fr:
            lines = fr.readlines()
        print(curLine(), raw_file_name, len(lines), "\n")

        save_file_name = "%s.txt" % corpus_type
        save_file = os.path.join(save_folder, save_file_name)
        with open(save_file, "w") as writeFile:
            for corpus_str in lines:
                corpus_map = json.loads(corpus_str.strip())
                passages = "".join(corpus_map['passages'])
                answer = "".join(corpus_map["answers"])
                question = corpus_map["question"]
                if len(passages)<10 or len(answer)<1 or len(question)<2 or len(passages)>800 or "________________" in passages:
                    # print(curLine(), len(corpus_map), corpus_map.keys(), "corpus_map:", corpus_map)
                    print(curLine(), "ignore answer:", answer, "question:", question, "passages:", passages)
                    # input(curLine())
                    continue
                try:
                    lcs1 = _compute_lcs(passages, answer)
                    lcs_rate = len(lcs1) / float(len(passages) + len(answer))
                except Exception as error:
                    print(curLine(), "answer:", answer, "passages:", passages)
                    print(curLine(), error)
                    continue
                length_abstract_list.append(len(passages))
                length_article_list.append(len(answer))
                line = "%s\t%s\t%s\t%f\n" % (question, passages, answer, lcs_rate)
                writeFile.write(line)
        print(curLine(), "save to %s" % save_file)
    all_num = len(length_article_list)
    print(
        "%slength_article max=%f ave=%f" % (
            curLine(), max(length_article_list), sum(length_article_list) / all_num))
    print(
        "%slength_abstract max=%f ave=%f" % (
            curLine(), max(length_abstract_list), sum(length_abstract_list) / all_num))
    print(curLine(), "all_num=%d" % all_num)


if __name__ == "__main__":
    host_name = "cloudminds"
    corpus_name = "ms_google"
    # corpus_folder = "/home/%s/PycharmProjects/kubeflow_project/qbsummary/data/ms_chinese/" % (host_name, corpus_name)
    corpus_folder = "/home/%s/Mywork/corpus/qqrec_data/ms_chinese/%s" % (host_name, corpus_name)
    save_folder = "/home/%s/Mywork/corpus/summary/%s" % (host_name, corpus_name)
    type_list = ["dev", "train"]
    process(corpus_folder, save_folder, type_list)
