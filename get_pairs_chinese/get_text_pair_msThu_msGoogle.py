# coding=utf-8
# 以生成式阅读理解的语料为数据集，训练摘要模型
import os, json
import sys

sys.path.append("..")
from compute_lcs import _compute_lcs
from curLine_file import curLine


def process(corpus_folder, save_folder, type_list, corpus_name_list):
    length_abstract_list = []
    length_article_list = []
    for corpus_type in type_list:
        save_file_name = "%s.txt" % corpus_type
        save_file = os.path.join(save_folder, save_file_name)
        with open(save_file, "w") as writeFile:
            for corpus_name in corpus_name_list:
                raw_file_name = "%s_ch.json" % corpus_type
                raw_file = os.path.join(corpus_folder, corpus_name, raw_file_name)
                with open(raw_file, "r") as fr:
                    lines = fr.readlines()
                print(curLine(), raw_file_name, len(lines), "\n")
                for corpus_str in lines:
                    corpus_map = json.loads(corpus_str.strip())
                    if "ms_thu" in corpus_name:
                        passages = corpus_map['context']
                        answer = corpus_map["answer"]
                    else:
                        passages = corpus_map['passages']
                        answer = corpus_map["answers"]

                    question = corpus_map["question"]
                    if len(passages)<10 or len(answer)<1 or len(question)<2:
                        # print(curLine(), "ignore answer:", answer, "question:", question, "passages:", passages)
                        continue
                    try:#  极个别情况下，计算ＬＣＳ时会陷入死循环，达到递归的最大限度
                        lcs1 = _compute_lcs(passages, answer)
                        lcs_rate = len(lcs1) / float(len(passages) + len(answer))
                    except Exception as error:
                        print(curLine(), "answer:", answer, "passages:", passages)
                        print(curLine(), error)
                        continue
                    length_abstract_list.append(len(answer))
                    length_article_list.append(len(passages))
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
    corpus_name_list = ["ms_thu", "ms_google"]
    corpus_folder = "/home/%s/Mywork/corpus/qqrec_data/ms_chinese" % (host_name)
    save_folder = "/home/%s/Mywork/corpus/summary/ms_thu_google" % (host_name)
    type_list = ["dev", "train"]
    process(corpus_folder, save_folder, type_list, corpus_name_list)
