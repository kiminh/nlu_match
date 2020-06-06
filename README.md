# LaserTagger
bash run.sh host_name

## 环境
Python 3.5+     python3.5~3.7版本下亲测可用，其他版本不清楚  
tensorflow1.15版本下亲测可用，其他版本不清楚 
both cpu or gpu is ok!  


一．场景和概述


二．模型介绍
谷歌在文献《Encode, Tag, Realize: High-Precision Text Editing》中采用序列标注的框架进行文本编辑（删除，插入，或替换一个token，token是模型处理文本的最小单元)，在文本拆分和自动摘要任务上取得了最佳效果。  
在同样采用BERT作为编码器的条件下，本方法相比于Seq2Seq的方法具有更高的可靠度，更快的训练和推理效率，且在语料规模较小的情况下优势更明显。
谷歌公开了本文献对应的代码，但是原有任务与当前任务有一定的差异性，需要修改部分代码，主要修改如下：  
A.分词方式：  
采用word_piece的分词方式将一句话分成若干token。  
对于汉字即以字为单位，对于英文和数字，是以亚词ｓｕｂ－ｗｏｒｄ为单位。这样既不容易产生ＯＯＶ，又能缩短输入的长度。  
B.推理效率：  
原代码每次只对一个输入进行摘要，改成每次对batch_size个文本进行复述，推理效率提高6倍。  
在我的台式机上可以设置batch_size＝６４
C.为兼顾效果与效率，将上图中原来的Encoder ＢＥＲＴ模块换成更为轻量级的RoBERT_Tiny_clue模块。  

对于当前的文本摘要服务，模型主要对token进行删除操作，偶尔会在说法不通顺时采用插入或替换操作。  

三.文件说明和实验步骤  
1.安装python模块
参见"requirements.txt", "start.sh"
2.下载预训练模型
考虑模型推理的效率，目前本项目采用RoBERTa-tiny-clue（中文版）预训练模型。
由于目前网络上有不同版本，现将本项目使用的预训练模型上传的百度网盘。链接: https://pan.baidu.com/s/1yho8ihR9C6rBbY-IJjSagA 提取码: 2a97
如果想采用其他预训练模型，请修改“configs/lasertagger_config.json".
3.训练和评测模型  
根据自己情况修改脚本"rephrase.sh"中2个文件夹的路径，然后运行  sh rephrase.sh
脚本中的变量HOST_NAME是作者为了方便设定路径使用的，请根据自己情况修改；  
如果只是离线的对文本进行批量的泛化，可以注释脚本中其他部分，只用predict_main.py就可以满足需求。  
4.启动文本复述服务  根据自己需要，可选  
根据自己情况修改"rephrase_server.sh"文件中几个文件夹的路径，使用命令"sh rephrase_server.sh"可以启动一个文本复述的API服务  
本API服务可以接收一个http的POST请求，解析并对其中的文本进行泛化，具体接口请看“rephrase_server/rephrase_server_flask.py"  

文本复述的语料需要自己整理语义一致的文本对。如果用自己业务场景下的语料最好，当然数量不能太少，如果没有或不够就加上LCQMC等语料中的正例。
然后用最长公共子串的长度限制一下，因为这个方法要求source和target的字面表达不能差异过大，可以参考一下“get_text_pair_lcqmc.py”。
目前，我的train.txt,tune.txt中都是三列即text1,text2,lcs_score,之间用tab"\t"分割。



## How to Cite LaserTagger

```
@inproceedings{malmi2019lasertagger,
  title={Encode, Tag, Realize: High-Precision Text Editing},
  author={Eric Malmi and Sebastian Krause and Sascha Rothe and Daniil Mirylenka and Aliaksei Severyn},
  booktitle={EMNLP-IJCNLP},
  year={2019}
}
```

## License

Apache 2.0; see [LICENSE](LICENSE) for details.
