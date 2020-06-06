# coding=utf-8
# 处理运营标注的摘要内容作为验证集
import os, json
import sys
import xlrd

sys.path.append("..")
from compute_lcs import _compute_lcs
from curLine_file import curLine, normal_transformer


def process(corpus_folder, save_folder):
    length_abstract_list = []
    length_article_list = []
    # for corpus_type in type_list:
    raw_file_name = "内容标注-10935.xls"
    raw_file = os.path.join(corpus_folder, raw_file_name)
    try:
        all_entity = xlrd.open_workbook(raw_file)
    except:
        print("%s %s not exist\n" % (curLine(), raw_file))

    try:
        sheet_str = "Sheet1"
        all_entity = all_entity.sheet_by_name(sheet_str)
    except:
        print("%s no sheet in %s named %s" % (curLine(), raw_file, sheet_str))
    row_num = all_entity.nrows  # 获取行数
    print(curLine(), raw_file_name, row_num, "\n")

    save_file_name = "yunying_annotate.txt"
    save_file = os.path.join(save_folder, save_file_name)
    with open(save_file, "w") as writeFile:
        # for corpus_str in lines:
        #     corpus_map = json.loads(corpus_str.strip())
        for i_row in range(1, row_num):
            tag = int(all_entity.cell_value(i_row, 4))
            assert tag in [1,2,3,4], "tag:%d" % tag
            question_list = all_entity.cell_value(i_row, 0).strip().split("&&")
            passages = str(all_entity.cell_value(i_row, 1)).strip()  #  SV导出的
            beizhu = str(all_entity.cell_value(i_row, 5)).strip()  # 个别情况，运营会备注出问题
            if len(beizhu) > 0:
                continue  # 备注的一般都是有问题的
            answer = str(all_entity.cell_value(i_row, 2)).strip()
            question = question_list[0]  # TODO min max
            for q in question_list:
                if len(q)>1 and len(q) < len(question):
                    question = q

            if tag not in [2,4]:  # 不需要简化（原答案在100字左右的）
                continue
            passages = normal_transformer(passages)
            if "\n" in passages:
                print(curLine(), passages)
                input(curLine())
            answer = normal_transformer(answer)
            question = normal_transformer(question)
            if len(passages)<10 or len(answer)<1 or len(question)<2 or len(answer) >= len(passages):
                print(curLine(), "ignore answer:", answer, "question:", question, "passages:", passages)
                # input(curLine())
                continue

            try: #  极个别情况下，计算ＬＣＳ时会陷入死循环，达到递归的最大限度
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
    corpus_name = "yunying"
    # corpus_folder = "/home/%s/PycharmProjects/kubeflow_project/qbsummary/data/ms_chinese/" % (host_name, corpus_name)
    corpus_folder = "/home/%s/Mywork/corpus/summary/%s" % (host_name, corpus_name)
    save_folder = "/home/%s/Mywork/corpus/summary/%s" % (host_name, corpus_name)
    process(corpus_folder, save_folder)
