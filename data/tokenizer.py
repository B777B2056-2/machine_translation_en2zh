#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import pickle
from typing import List
from sklearn.feature_extraction.text import CountVectorizer


class Tokenizer:
    """将文本转换为词向量：读取文本 -> tokenized分词 -> token编码：转成input ids向量"""
    SENTENCE_START_PLACEHOLDER = "<sos>"  # 句子起始符
    SENTENCE_END_PLACEHOLDER = "<eos>"  # 句子结束符

    def __init__(self, file_path: str = ""):
        """读取文本"""
        if file_path != "":
            with open(file_path, "r", encoding='UTF-8') as f:
                self.text = f.readlines()

        self.__word2idx = {}
        self.__idx2word = {}
        self.sentences_num, self.__start_symbol_idx, self.__end_symbol_idx = 0, -1, -1

    @staticmethod
    def __tokenized(language: str, lines: List[str]) -> List[str]:
        """文本分词，例如：单句为 1929 or 1989? -> 分词后 [[1929, or, 1989, ?]]"""
        import jieba
        import nltk
        # nltk.download('punkt_tab')

        assert language in ('en', 'zh') and len(lines) != 0

        corpus = []
        for line in lines:
            sentence = [Tokenizer.SENTENCE_START_PLACEHOLDER]
            words = nltk.word_tokenize(line) if language == "en" else jieba.lcut(line)
            for word in words:
                sentence.append(word)
            sentence.append(Tokenizer.SENTENCE_END_PLACEHOLDER)
            corpus.append(" ".join(sentence))
        return corpus

    def __word_to_idx(self, corpus: List[str]) -> List[List[int]]:
        """将分词产出的token，转换为数值向量（即词索引）"""
        self.vectorizer = CountVectorizer(lowercase=False)
        _ = self.vectorizer.fit_transform(corpus)
        vocab = self.vectorizer.vocabulary_
        self.__word2idx = vocab.copy()
        for (word, idx) in self.__word2idx.items():
            self.__idx2word[idx] = word
        tokenized_sentences = []
        for doc in corpus:
            tokens = self.vectorizer.build_analyzer()(doc)
            indices = [self.__word2idx[token] for token in tokens if token in self.__word2idx]
            tokenized_sentences.append(indices)
        # 获取起始符、结束符索引
        self.__start_symbol_idx = self.__word2idx[Tokenizer.SENTENCE_START_PLACEHOLDER.lstrip("<").rstrip(">")]
        self.__end_symbol_idx = self.__word2idx[Tokenizer.SENTENCE_END_PLACEHOLDER.lstrip("<").rstrip(">")]
        return tokenized_sentences

    def __build_vocab(self, language: str) -> List[List[int]]:
        """构建词表（训练前进行）"""
        # 分词
        corpus = self.__tokenized(language=language, lines=self.text)
        self.sentences_num = len(corpus)
        # token编码：转成input ids向量
        return self.__word_to_idx(corpus)

    def tokenize(self, language: str, prompts=None) -> List[List[int]]:
        # 判断是否已有词表，若无则构建后返回
        if self.sentences_num == 0:
            return self.__build_vocab(language)
        # 已有词表，分词后直接查表，并添加起始、结束符索引位置
        self.text = prompts
        corpus = self.__tokenized(language=language, lines=self.text)  # 分词
        vector = []
        for doc in corpus:
            tokens = self.vectorizer.build_analyzer()(doc)
            indices = [self.__word2idx[token] for token in tokens]
            vector.append(indices)
        return vector

    def detokenize(self, tokens: List[int]) -> List[str]:
        """从传入token反向构建词"""
        words = []
        for word_idx in tokens:
            words.append(self.__idx2word[word_idx])
        return words

    def start_flag_id(self) -> int:
        return self.__start_symbol_idx

    def end_flag_id(self) -> int:
        return self.__end_symbol_idx

    def save(self, language:str, dir_path:str= "output/vocabs") -> None:
        """保存词表（只保存一次）"""
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if "vectorizer" in f and language in f and f.endswith(".pkl")]
            if len(files) != 0:
                return
        else:
            os.makedirs(dir_path, exist_ok=True)
        with open(os.path.join(dir_path, f"{language}_vectorizer_{self.sentences_num}.pkl"), 'wb') as f:
            pickle.dump(self.vectorizer.vocabulary_, f)

    @classmethod
    def load_from_disk(cls, language: str, dir_path:str="output/vocabs") -> "Tokenizer":
        """从文件里恢复词表"""
        files = [f for f in os.listdir(dir_path) if "vectorizer" in f and language in f and f.endswith(".pkl")]
        files.sort(key=lambda x: x.split("_")[-1].split(".")[0], reverse=True)
        tgt_file_path = os.path.join(dir_path, files[0])
        vectorizer = CountVectorizer(vocabulary=pickle.load(open(tgt_file_path,'rb')))
        sentences_num = int(files[0].split(sep="_")[1])
        instance = cls()
        instance.vectorizer = vectorizer
        instance.sentences_num = sentences_num
        return instance
