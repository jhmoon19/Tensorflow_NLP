import os
import re
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from konlpy.tag import Okt

# print("Hello")

FILTERS = "([~.,!?\"':;)(])"
PAD = "<PAD>" # 패딩 토큰
STD = "<SOS>" # Start Of Sentence (시작 토큰)
END = "<END>" # 종료 토큰
UNK = "<UNK>" # 사전에 없는 단어 토큰

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

MAX_SEQUENCE = 25

def load_data(path):
    # 판다스로 데이터 불러오기
    data_df = pd.read_csv(path, encoding='utf8', header=0)
    question, answer = list(data_df['Q']), list(data_df['A'])
    
    return question, answer

def data_tokenizer(data):
    # 토크나이징 해서 담을 배열 생성
    words = []
    for sentence in data:
        # CHANGE_FILTER.sub('', sentence) 와 동일! 
        setence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split():
            words.append(word)
        
    return [word for word in words if word]

def prepro_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = list()
    
    for seq in tqdm(data):
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)
        
    return result_data  # ["_ _ _", "_ _ _", ...]

def load_vocabulary(path, vocab_path, tokenize_as_morph=False):
    # 사전을 담을 배열 준비
    vocabulary_list = []
    
    if not os.path.exists(vocab_path):
        # 데이터 파일이 존재한다면 (--> 사전 만듦)
        if os.path.exists(path):
            question, answer = load_data(path)
            
            # 형태소에 따른 토크나이저 처리
            if tokenize_as_morph:
                question = prepro_like_morphlized(question)
                answer = prepro_like_morphlized(answer)
                
            data = []
            data.extend(question)
            data.extend(answer)
            
            words = data_tokenizer(data)
            words = list(set(words))
            
            # 토큰 리스트 앞부분에 MARKER들 넣어주기
            words[:0] = MARKER
        
        # 사전을 리스트로 만들었으니, 이걸 사전 파일 만들어 넣음
        with open(vocab_path, 'w', encoding='utf8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')
                
    # 사전 파일이 존재하면 여기에서
    # 그 파일을 불러서 배열에 넣어 준다.
    with open(vocab_path, 'r', encoding='utf8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())
            
    # 배열에 내용을 키와 값이 있는 딕셔너리 구조로 만듦.
    char2idx, idx2char = make_vocabulary(vocabulary_list)
    
    return char2idx, idx2char, len(char2idx)

def make_vocabulary(vocabulary_list):
    # 간단히 enumerate 함수로 단어 정수 매핑
    char2idx = {char:idx for idx, char in enumerate(vocabulary_list)}
    idx2char = {idx:char for idx, char in enumerate(vocabulary_list)}
    return char2idx, idx2char

# 인코더 부분 입력 전처리 함수
def enc_processing(value, dictionary, tokenize_as_morph=False):
    # 인코딩되는 한 문장의 인덱스 리스트 (누적됨)
    sequences_input_index = []
    # 인코딩되는 한 문장의 길이 리스트 (누적됨)
    sequences_length = []
    
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)
        
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, '', sequence)
        sequence_index = []
        
        for word in sequence.split():
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            else:
                sequence_index.extend([dictionary[UNK]])
        # 인코딩할 해당 문장의 토큰별 정수매핑 결과가 
        # max seq 길이 초과한다면, 그 길이까지 자르기
        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]
        
        # 패딩 이전의 해당 문장 시퀀스 길이 저장
        sequences_length.append(len(sequence_index))
        # max seq 길이 남는 부분만큼 "<PAD>" 로 채워넣기!
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        # 패딩까지 완료된 최종 시퀀스 인덱스 추가
        sequences_input_index.append(sequence_index)
    
    """ 
    np.asarray: 원본 변경 시 복사본까지 변경
    np.array: 원본 변경 시 복사본 변경 안됨!
    """
    return np.asarray(sequences_input_index), sequences_length

# 디코더의 입력값 전처리 함수 ("<SOS>" 포함)
def dec_output_processing(value, dictionary, tokenize_as_morph=False):
    sequences_output_index = []
    sequences_length = []
    
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)
        
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, '', sequence)
        sequence_index = []
        
        # 시작부분에 "<SOS>" 추가 
        sequence_index = [dictionary[STD]] + [dictionary[word] if word in dictionary 
                                              else dictionary[UNK] for word in sequence.split()]
        
        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence.index[:MAX_SEQUENCE]
            
        sequences_length.append(len(sequence_index))
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_output_index.append(sequence_index)
        
    return np.asarray(sequences_output_index), sequences_length

# 디코더의 타깃값 전처리 함수 ("<END>" 포함)
def dec_target_processing(value, dictionary, tokenize_as_morph=False):
    sequences_target_index = []
    
    if tokenize_as_morph:
        value = prepro_like_morphlized(value)
        
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, '', sequence)
        sequence_index = [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]
        
        if len(sequence_index) >= MAX_SEQUENCE:
            # "<END>" 넣을 자리는 빼고 max_seq 길이만큼 자르기
            sequence_index = sequence_index[:MAX_SEQUENCE-1] + [dictionary[END]]
        else:
            # 마지막에 "<END>" 추가
            sequence_index += [dictionary[END]]
            
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        
        sequences_target_index.append(sequence_index)
        
    return np.asarray(sequences_target_index)