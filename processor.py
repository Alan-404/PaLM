import re
import pickle
import os
import numpy as np
from tqdm import tqdm

class PaLMProcessor:
    def __init__(self, pretrained: str = None, special_tokens: list = [], info_tokens: dict = {}, padding_token: str = "<pad>") -> None:
        init_tokens = ["<pad>", "<start>", "<end>", "<oov>", "<sep>"]
        self.cleaner = Cleaner()
        self.vocab_dict = dict()
        self.dictionary = []
        self.info = info_tokens

        self.word_break_icon = "</w>"

        self.original_size = 0
        self.epoch = 0

        self.pretrained = pretrained
        if self.pretrained is not None and os.path.exists(self.pretrained) == True:
            self.load_tokenizer(self.pretrained)
        else:
            self.special_tokens = init_tokens + special_tokens
            for key in info_tokens.keys():
                self.special_tokens.append(key)

        self.padding_token = self.dictionary.index(padding_token)

    def get_special_token(self, name: str):
        name = f"<{name}>"
        assert name in self.dictionary

        return self.dictionary.index(name)
    
    def __save_tokenzier(self, path: str):
        obj = {
            TokenizerInfo.DICTIONARY: self.dictionary,
            TokenizerInfo.VOCABULARY: self.vocab_dict,
            TokenizerInfo.SPECIAL_TOKENS: self.special_tokens,
            TokenizerInfo.INFO_TOKENS: self.info,
            TokenizerInfo.ORIGINAL_SIZE: self.original_size,
            TokenizerInfo.EPOCH: self.epoch
        }
        with open(path, 'wb') as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    def save_tokenizer(self, path: str):
        try:
            self.__save_tokenzier(path)
        except Exception as e:
            print(str(e))
            print("Tokenizer is saved at root project.")
            self.__save_tokenzier("./tokenizer.pkl")

    def load_tokenizer(self, path: str):
        if os.path.exists(path):
            with open(path, 'rb') as file:
                data = pickle.load(file)
            self.dictionary = data[TokenizerInfo.DICTIONARY]
            self.vocab_dict = data[TokenizerInfo.VOCABULARY]
            self.special_tokens = data[TokenizerInfo.SPECIAL_TOKENS]
            self.original_size = data[TokenizerInfo.ORIGINAL_SIZE]
            self.info = data[TokenizerInfo.INFO_TOKENS]
            self.epoch = data[TokenizerInfo.EPOCH]

    def cal_total_vocab(self, data: list):
        dictionary = []
        for item in data:
            seq = self.cleaner.clean(item)
            words = seq.split(" ")
            for word in words:
                if word not in dictionary:
                    dictionary.append(word)
        return len(dictionary)
    
    def init_vocab_dict(self, data: list):
        vocab_dict = dict()
        dictionary = []
        for seq in data:
            seq = self.cleaner.clean(seq)
            words = str(seq).split(" ")
            for word in words:
                if word not in dictionary:
                    dictionary.append(word)
                temp = []
                if word in self.special_tokens or word in self.cleaner.puncs:
                    temp.append(word)
                else:
                    for char in word:
                        temp.append(char)
                    temp.append(self.word_break_icon)
                if tuple(temp) not in vocab_dict:
                    vocab_dict[tuple(temp)] = 1
                else:
                    vocab_dict[tuple(temp)] += 1
        return vocab_dict, len(dictionary)
    
    def create_dictionary(self, dictionary: dict):
        tokens = []
        for token in self.special_tokens:
            tokens.append(token)
        for item in dictionary:
            for char in item:
                if char not in tokens:
                    tokens.append(char)
        return tokens

    def create_pair(self, vocab_dict: dict):
        pair = dict()
        for vocab in vocab_dict:
            for index in range(len(vocab)-1):
                if (vocab[index], vocab[index+1]) not in pair:
                    pair[(vocab[index], vocab[index+1])] = 1
                else:
                    pair[(vocab[index], vocab[index+1])] += 1
        return pair

    def fit(self, data: list, max_iterations: int = 10, sigma: float = 2.0):
        if self.original_size == 0 and len(self.dictionary) == 0:
            self.vocab_dict, self.original_size = self.init_vocab_dict(data)
            self.dictionary = self.create_dictionary(self.vocab_dict)
        print(f"Original Dictionary Size: {self.original_size}")
        print("========== Training Tokenizer ============")
        for _ in tqdm(range(max_iterations)):
            pairs = self.create_pair(self.vocab_dict)
            max_item = max(pairs, key=lambda k: pairs[k])
            temp_dict = dict()
            for vocab in self.vocab_dict:
                temp = []
                flag = False
                for index in range(len(vocab)):
                    if flag is True:
                        flag = False
                        continue
                    if vocab[index] != max_item[0]:
                        temp.append(vocab[index])
                    else:
                        if index == len(vocab)-1:
                            temp.append(vocab[index])
                            continue
                        if vocab[index + 1] == max_item[1]:
                            temp.append(vocab[index] + vocab[index+1])
                            flag = True
                        else:
                            temp.append(vocab[index])
                temp_dict[tuple(temp)] = self.vocab_dict[vocab]
            
            self.vocab_dict = temp_dict
            self.dictionary = self.create_dictionary(self.vocab_dict)

            self.epoch += 1
            if len(self.dictionary) >= int(self.original_size/sigma) and len(self.dictionary) < self.original_size:
                break
        print(f"Epoch {self.epoch+1} Dictionary Size: {len(self.dictionary)}")
    
    def find(self, word: str, special_token: bool = False):
        text = [*word]
        if special_token == False:
            text += [self.word_break_icon]
        else:
            return [self.dictionary.index(word)]
        embedding = []
        mixed = 0
        for index in range(len(text)):
            if mixed > index:
                continue
            if mixed >= len(text):
                break
            subseq = len(text[index:])
            for i in range(subseq):
                pattern = "".join(text[index:subseq-i + index])
                if pattern in self.dictionary:
                    embedding.append(self.dictionary.index(pattern))
                    mixed += len(text[index:subseq - i + index])
                    break
        return embedding
    
    def padding_sequence(self, sequence, padding: str, maxlen: int) -> np.ndarray:
        delta = maxlen - len(sequence)
        zeros = np.zeros(delta, dtype=np.int64)

        if padding.strip().lower() == 'post':
            return np.concatenate((sequence, zeros), axis=0)
        elif padding.strip().lower() == 'pre':
            return np.concatenate((zeros, sequence), axis=0)

    def truncating_sequence(self, sequence, truncating: str, maxlen: int) -> np.ndarray:
        if truncating.strip().lower() == 'post':
            return sequence[0:maxlen]
        elif truncating.strip().lower() == 'pre':
            delta = sequence.shape[0] - maxlen
            return sequence[delta: len(sequence)]

    def pad_sequences(self, sequences: list, maxlen: int, padding: str = 'post', truncating: str = 'post') -> np.ndarray:
        result = []
        for _, sequence in enumerate(sequences):
            delta = sequence.shape[0] - maxlen
            if delta < 0:
                sequence = self.padding_sequence(sequence, padding, maxlen)
            elif delta > 0:
                sequence = self.truncating_sequence(sequence, truncating, maxlen)
            result.append(sequence)
        
        return np.array(result)
    
    def text_to_sequences(self, data: list, max_length: int = None, start_token: bool = False, end_token: bool = False, sep_token: bool = False):
        digits = []
        maxlen = 0
        for seq in data:
            seq = self.cleaner.clean(seq)
            words = seq.split(" ")
            temp = []
            if start_token:
                temp += [self.get_special_token("start")]
            for word in words:
                special_token = word in self.special_tokens
                digit_word = self.find(word, special_token)
                temp += digit_word
            if sep_token:
                temp += [self.get_special_token("sep")]
            elif end_token:
                temp += [self.get_special_token("end")]
            if maxlen < len(temp):
                maxlen = len(temp)
            digits.append(np.array(temp))
        if max_length is None:
            padded_data = self.pad_sequences(digits, maxlen=maxlen)
        else:
            padded_data = self.pad_sequences(digits, max_length)
        return padded_data
    
    def text2sequence(self, input_sample: str, output_sample: str):
        return self.text2digit(f"{input_sample} <sep> {output_sample}", start_token=True, end_token=True)
        

    def text2digit(self, sequence: str, start_token: bool = False, sep_token: bool = False, end_token: bool = False):
        digits = []
        sequence = self.cleaner.clean(sequence)

        words = sequence.split(" ")
        if start_token:
            digits += [self.get_special_token("start")]
        for word in words:
            special_token = word in self.special_tokens
            digit_word = self.find(word, special_token)
            digits += digit_word
        if sep_token:
            digits += [self.get_special_token("sep")]
        elif end_token:
            digits += [self.get_special_token("end")]

        return np.array(digits)
    
    def save_data(self, data: np.ndarray, path: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    def decode(self, tokens: np.ndarray):
        text = []
        for token in tokens:
            word = self.dictionary[token]
            if word in self.info:
                text.append(str(self.info[word]))
            elif word == self.word_break_icon:
                text.append(" ")
            else:
                text.append(self.dictionary[token])
        
        return "".join(text).replace("</w>", " ")
    
    def get_data(self, path: str) -> np.ndarray:
        with open(path, 'rb') as file:
            return pickle.load(file)

class Cleaner:
    def __init__(self, puncs: list = r"([:./,?!@#$%^&=`~*\(\)\[\]\"\'\-\\])") -> None:
        self.puncs = puncs
    def clean(self, seq: str):
        seq = re.sub(self.puncs, r" \1 ", seq)
        seq = seq.strip()
        seq = re.sub("\s\s+", " ", seq)
        seq = seq.lower()
        return seq


def load_data(path):
    if os.path.exists(path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data
    return None

def save_data(path: str, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


class TokenizerInfo:
    DICTIONARY = 'dictionary'
    VOCABULARY = 'vocabulary'
    SPECIAL_TOKENS = 'special_tokens'
    INFO_TOKENS = 'info_tokens'
    ORIGINAL_SIZE = 'original_size'
    EPOCH = 'epoch'
