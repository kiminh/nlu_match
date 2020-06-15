import os
from collections import defaultdict
from curLine_file import curLine
import json, time
import re
from curLine_file import curLine

re_phoneNum = re.compile("[0-9一二三四五六七八九十拾]+")  # 编译

waibu_folder = "/home/cloudminds/Mywork/corpus/ner_corpus/music_corpus/music_entity/"
ignoreSongMap = {}
frequentSong = {}
frequentSinger = {}
with open(waibu_folder+"ignoreSongMap.json", "r") as f:
    ignoreSongMap = json.load(f)
with open(waibu_folder+"frequentSong.json", "r") as f:
    frequentSong = json.load(f)

with open(waibu_folder+"frequentSinger.json", "r") as f:
    frequentSinger = json.load(f)
neibu_folder ="/home/cloudminds/Mywork/corpus/compe/69"

class State(object):
    __slots__ = ['identifier', 'symbol', 'success', 'transitions', 'parent',
                 'matched_keyword', 'longest_strict_suffix', 'meta_data']

    def __init__(self, identifier, symbol=None, parent=None, success=False):
        self.symbol = symbol
        self.identifier = identifier
        self.transitions = {}
        self.parent = parent
        self.success = success
        self.matched_keyword = None
        self.longest_strict_suffix = None


class Result(object):
    __slots__ = ['keyword', 'location', 'meta_data']

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return_str = ''
        for k in self.__slots__:
            return_str += '{}:{:<20}\t'.format(k, json.dumps(getattr(self, k)))
        return return_str

class KeywordTree(object):
    def __init__(self, case_insensitive=True):
        '''
        @param case_insensitive: If true, case will be ignored when searching.
                                 Setting this to true will have a positive
                                 impact on performance.
                                 Defaults to false.
        '''
        self._zero_state = State(0)
        self._counter = 1
        self._finalized = False
        self._case_insensitive = case_insensitive

    def add(self, keywords, meta_data=None):
        '''
        Add a keyword to the tree.
        Can only be used before finalize() has been called.
        Keyword should be str or unicode.
        '''
        if self._finalized:
            raise ValueError('KeywordTree has been finalized.' +
                             ' No more keyword additions allowed')
        original_keyword = keywords
        if self._case_insensitive:
            if isinstance(keywords, list):
                keywords = map(str.lower, keywords)
            elif isinstance(keywords, str):
                keywords = keywords.lower()
                # if keywords != original_keyword:
                #     print(curLine(), keywords, original_keyword)
                #     input(curLine())
            else:
                raise Exception('keywords error')
        if len(keywords) <= 0:
            return
        current_state = self._zero_state
        for word in keywords:
            try:
                current_state = current_state.transitions[word]
            except KeyError:
                next_state = State(self._counter, parent=current_state,
                                   symbol=word)
                self._counter += 1
                current_state.transitions[word] = next_state
                current_state = next_state
        current_state.success = True
        current_state.matched_keyword = original_keyword
        current_state.meta_data = meta_data

    def search(self, text, greedy=False, cut_word=False, cut_separator=' '):
        '''

        :param text:
        :param greedy:
        :param cut_word:
        :param cut_separator:
        :return:
        '''
        gen = self._search(text, cut_word, cut_separator)
        pre = None
        for result in gen:
            assert isinstance(result, Result)
            if not greedy:
                yield result
                continue
            if pre is None:
                pre = result

            if result.location > pre.location:
                yield pre
                pre = result
                continue

            if len(result.keyword) > len(pre.keyword):
                pre = result
                continue
        if pre is not None:
            yield pre

    def _search(self, text, cut_word=False, cut_separator=' '):
        '''
        Search a text for all occurences of the added keywords.
        Can only be called after finalized() has been called.
        O(n) with n = len(text)
        @return: Generator used to iterate over the results.
                 Or None if no keyword was found in the text.
        '''
        # if not self._finalized:
        #     raise ValueError('KeywordTree has not been finalized.' +
        #                      ' No search allowed. Call finalize() first.')
        if self._case_insensitive:
            if isinstance(text, list):
                text = map(str.lower, text)
            elif isinstance(text, str):
                text = text.lower()
            else:
                raise Exception('context type error')

        if cut_word:
            if isinstance(text, str):
                text = text.split(cut_separator)

        current_state = self._zero_state
        for idx, symbol in enumerate(text):
            current_state = current_state.transitions.get(
                symbol, self._zero_state.transitions.get(symbol, self._zero_state))
            state = current_state
            while state != self._zero_state:
                if state.success:
                    keyword = state.matched_keyword
                    yield Result(**{
                        'keyword': keyword,
                        'location': idx - len(keyword) + 1,
                        'meta_data': state.meta_data
                    })
                    # yield (keyword, idx - len(keyword) + 1, state.meta_data)
                state = state.longest_strict_suffix

    def finalize(self):
        '''
        Needs to be called after all keywords have been added and
        before any searching is performed.
        '''
        if self._finalized:
            raise ValueError('KeywordTree has already been finalized.')
        self._zero_state.longest_strict_suffix = self._zero_state
        processed = set()
        to_process = [self._zero_state]
        while to_process:
            state = to_process.pop()  # 删除并返回最后一个元素，所以这是深度优先搜索
            processed.add(state.identifier)
            for child in state.transitions.values():
                if child.identifier not in processed:
                    self.search_lss(child)
                    to_process.append(child)
        self._finalized = True

    def __str__(self):
        return "ahocorapy KeywordTree"


    def search_lss(self, state):
        if state.longest_strict_suffix is None:
            parent = state.parent
            traversed = parent.longest_strict_suffix
            while True:
                if state.symbol in traversed.transitions and \
                                traversed.transitions[state.symbol] != state:
                    state.longest_strict_suffix = \
                        traversed.transitions[state.symbol]
                    break
                elif traversed == self._zero_state:
                    state.longest_strict_suffix = self._zero_state
                    break
                else:
                    traversed = traversed.longest_strict_suffix
            suffix = state.longest_strict_suffix
            if suffix.longest_strict_suffix is None:
                self.search_lss(suffix)
            for symbol, next_state in suffix.transitions.items():
                if (symbol not in state.transitions and
                            suffix != self._zero_state):
                    state.transitions[symbol] = next_state

def add_to_ac(ac, entity_type, entity_before, entity_after, pri):
    flag = "ignore"
    if entity_type == "song" and ((entity_after in ignoreSongMap or entity_after in frequentSong) and entity_after not in {"李白"}):
        return flag
    if entity_type == "singer" and entity_after in frequentSinger:
        return flag
    elif entity_type == "toplist" and entity_before == "首张":
        return flag
    elif entity_type == "emotion" and entity_before in {"high歌","相思","喜欢"}:  # train和devs就是这么标注的
        return flag
    elif entity_type == "language" and entity_before in ["中国"]:  # train和devs就是这么标注的
        return flag
    ac.add(keywords=entity_before, meta_data=(entity_after,pri))
    return "add success"

# AC自动机, similar to trie tree
# 也许直接读取下载的ｘｌｓ文件更方便，但那样需要安装ｘｌｒｄ模块
entity_folder = "/home/cloudminds/Mywork/corpus/compe/69/slot-dictionaries"
domain2entity_map = {}
domain2entity_map["music"] = ["age", "singer", "song", "toplist", "theme", "style", "scene", "language", "emotion", "instrument"]
domain2entity_map["navigation"] = ["custom_destination", "destination", "origin"]
domain2entity_map["phone_call"] = ["phone_num", "contact_name"]
self_entity_trie_tree = {}  # 总的实体字典  自己建立的某些实体类型的实体树
for domain, entity_type_list in domain2entity_map.items():
    print(curLine(), domain, entity_type_list)
    for entity_type in entity_type_list:
        if entity_type not in self_entity_trie_tree:
            ac = KeywordTree(case_insensitive=True)
        else:
            ac = self_entity_trie_tree[entity_type]
        ############# 按照优先级别从低到高的顺序来加载
        # if entity_type in ["song"]: # "singer",
        #     entity_file = waibu_folder + "%s.json"%entity_type
        #     with open(entity_file, "r") as f:
        #         current_entity_dict = json.load(f)
        #         print(curLine(), "get %d %s from %s" %
        #               (len(current_entity_dict), entity_type, entity_file))
        #     for entity_before, entity_times in current_entity_dict.items():
        #         if entity_type=="song" and len(entity_before)<3:
        #             continue  # ignore
        #         entity_after = entity_before
        #         pri = 1
        #         if entity_type in ["song"]:
        #             pri -= 0.5
        #         add_to_ac(ac, entity_type, entity_before, entity_after, pri=pri)

        # 从标注语料中挖掘得到,最高优先级
        entity_file = os.path.join(neibu_folder, "%s.json" % entity_type)
        with open(entity_file, "r") as fr:
            current_entity_dict = json.load(fr)
        for entity_before, entity_after_times in current_entity_dict.items():
            entity_after = entity_after_times[0]
            pri = 2
            if entity_type in ["song"]:
                pri -= 0.5
            add_to_ac(ac, entity_type, entity_before, entity_after, pri=pri)

        # 给的实体库
        entity_file = os.path.join(entity_folder, "%s.txt" % entity_type)
        if os.path.exists(entity_file):
            with open(entity_file, "r") as fr:
                lines = fr.readlines()
            print(curLine(), "get %d %s from %s" % (len(lines), entity_type, entity_file))
            for line in lines:
                entity_after = line.strip()
                entity_before = entity_after # TODO
                pri = 3
                if entity_type in ["song"]:
                    pri -= 0.5
                add_to_ac(ac, entity_type, entity_before, entity_after, pri=pri)

        ac.finalize()
        self_entity_trie_tree[entity_type] = ac


def get_all_entity(corpus, useEntityTypeList):
    self_entityTypeMap = defaultdict(list)
    for entity_type in useEntityTypeList:
        result = self_entity_trie_tree[entity_type].search(corpus)
        for res in result:
            after, priority = res.meta_data
            self_entityTypeMap[entity_type].append({'before': res.keyword, 'after': after, "priority":priority})
    return self_entityTypeMap


def get_slot_info(query, domain):
    useEntityTypeList = domain2entity_map[domain]
    entityTypeMap = get_all_entity(query, useEntityTypeList=useEntityTypeList)
    if "phone_num" in useEntityTypeList:
        token_numbers = re_phoneNum.findall(query)
        for number in token_numbers:
            entityTypeMap["phone_num"].append({'before':number, 'after':number, 'priority': 2})
    # print(curLine(), "entityTypeMap", entityTypeMap)
    # for entity_type, entity_info_list in entityTypeMap.items():
    #     for entity_info in entity_info_list:
    #         entity_before = entity_info['before']
    #         priority = entity_info['priority']
    #         if len(entity_before) < 2 and entity_before not in ["家","妈"]:
    #             continue
    #         entity_map[entity_before] = (entity_type, entity_info['after'], priority) # TODO song的优先级应该低一点
    #         # if entity_before not in entity_map or (priority>entity_map[entity_before][2]):
    #         #     entity_map[entity_before] = (entity_type, entity_info['after'], priority)
    # print(curLine(), len(entity_map), "entity_map", entity_map)
    # if "phone_num" in useEntityTypeList:
    #     token_numbers = re_phoneNum.findall(query)
    #     for number in token_numbers:
    #         entity_map[number] = ("phone_num", number, 2)
    entity_list_all = [] #汇总所有实体
    for entity_type, entity_list in entityTypeMap.items():
        for entity in entity_list:
            entity_before = entity['before']
            if len(entity_before) < 2 and entity_before not in ["家","妈"]:
                continue
            entity_list_all.append((entity_type, entity_before, entity['after'], entity['priority']))
    entity_list_all = sorted(entity_list_all, key=lambda item: len(item[1])*100+item[3],
                             reverse=True)  # new_entity_map 中key是实体,value是实体类型
    slot_info = query
    exist_entityType_set = set()
    replace_mask = [0] * len(query)
    for entity_type, entity_before, entity_after, priority in entity_list_all:
        if entity_before not in query:
            continue
        if entity_type in exist_entityType_set:
            continue  # 已经有这个类型了,忽略 # TODO
        start_location = slot_info.find(entity_before)
        if start_location > -1: #  exist
            exist_entityType_set.add(entity_type)
            if entity_after == entity_before:
                entity_info_str = "<%s>%s</%s>" % (entity_type, entity_after, entity_type)
            else:
                entity_info_str = "<%s>%s||%s</%s>" % (entity_type, entity_before, entity_after, entity_type)
            slot_info = slot_info.replace(entity_before, entity_info_str)
            query = query.replace(entity_before, "")
        else:
            print(curLine(), replace_mask, slot_info, "entity_type:", entity_type, entity_before)
    return slot_info

if __name__ == '__main__':

    for query in ["拨打10086", "打电话给100十五", "打电话给一二三拾"]:
        res = get_slot_info(query, domain="phone_call")
        print(curLine(), query, res)

    for query in ["节奏来一首一朵鲜花送给心上人", "播放歌曲远走高飞"]:
        res = get_slot_info(query, domain="music")
        print(curLine(), query, res)