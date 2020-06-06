# coding=utf-8
# 分割语料
import os
import csv
from collections import defaultdict

from curLine_file import curLine, normal_transformer, other_tag


def process(source_file, train_file, dev_file):
    dev_lines = []
    train_num = 0
    intent_distribution = defaultdict(dict)
    with open(source_file, "r") as f:
        reader = csv.reader(f)
        with open(train_file, "w") as f_train:
            train_write = csv.writer(f_train, dialect='excel')
            for row_id, line in enumerate(reader):
                if row_id==0:
                    continue
                (sessionId, raw_query, domain_intent, param) = line
                if domain_intent == other_tag:
                    domain = other_tag
                    intent = other_tag
                else:
                    domain, intent = domain_intent.split(".")
                if intent in intent_distribution[domain]:
                    intent_distribution[domain][intent] += 1
                else:
                    intent_distribution[domain][intent] = 0
                if row_id == 0:
                    continue
                sessionId = int(sessionId)
                if sessionId % 10>0:
                    train_write.writerow(line)
                    train_num += 1
                else:
                    dev_lines.append(line)
    with open(dev_file, "w") as f_dev:
        write = csv.writer(f_dev, dialect='excel')
        for line in dev_lines:
            write.writerow(line)
    print(curLine(), "dev=%d, train=%d" % (len(dev_lines), train_num))
    # print(curLine(), "all_num=%d" % all_num)
    for domain, intent_num in intent_distribution.items():
        print(curLine(), domain, intent_num)

if __name__ == "__main__":
    host_name = "cloudminds"
    corpus_folder = "/home/%s/Mywork/corpus/compe/69" % (host_name)
    source_file = os.path.join(corpus_folder, "train.csv")
    train_file = os.path.join(corpus_folder, "train.txt")
    dev_file = os.path.join(corpus_folder, "dev.txt")
    process(source_file, train_file, dev_file)
