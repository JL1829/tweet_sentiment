"""
Author: Lu ZhiPing
Email: lu.zhiping@u.nus.edu
"""
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.tokenize import sent_tokenize, TweetTokenizer

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


class SimpleTokenizer:
    """
    Automatic to clen up the text and tokenize it.

    Example:
    >>> from sentiment.dataset.tokenizer import SimpleTokenizer
    >>> simple_text = 'Bukit Merah HDB branch closes temporarily after second employee tests positive for '\
        'COVID-19 https://t.co/yaMECYuNIN https://t.co/kz7BNBzjez '
    >>> tokenizer = SimpleTokenizer()
    >>> tokens = tokenizer(simple_text)
    >>> tokens
    ['bukit', 'merah', 'hdb', 'branch', 'close', 'temporarily', 'second', 'employee', 'test', 'positive', 'covid',
    '19', 'URL', 'URL']

    """

    def __init__(self):
        self.tokenizer_words = TweetTokenizer()
        self.stemmer = SnowballStemmer('english')
        self.stop = set(stopwords.words('english'))
        excluded_stops = ['no', 'not', 'what', 'who', 'how', 'where', 'when', 'has', 'have', 'do', 'does', 'should',
                          'would', 'can', 'could']
        [self.stop.remove(x) for x in excluded_stops if x in self.stop]
        self.exclude = set(string.punctuation)
        self.exclude.update([">", "<", "``", "”", "“", "’", "...", ",", "–", "—", ".", "/", "?", "!"])
        self.lemma = WordNetLemmatizer()

    @staticmethod
    def __is_email(s):
        if re.match("[\w.-_]+@[\w.-_]+", s):
            return True
        else:
            return False

    @staticmethod
    def __is_url_str(s):
        if re.match('(https?:)?//(?:[-\w.]|%[\da-fA-F]{2})+', s):
            return True
        else:
            return False

    @staticmethod
    def __is_time(s):
        if re.match("\d?\d\:\d\d", s):
            return True
        elif re.match("\d\d\.\d\d\.\d\d\d\d", s):
            return True
        else:
            return False

    @staticmethod
    def __remove_chars(s, chars=None, replace_with=" "):
        if chars is None:
            chars = [".", "/", "(", ")", "_", ":"]
        for x in chars:
            s = str(s).replace(x, replace_with)
        return s

    @staticmethod
    def __clean_word(s):
        if (s.startswith("#") or s.startswith("@")) and len(s) > 2:
            new = list(s)
            new[0] = ''
            s = ''.join(new)
        if s.endswith("-") and len(s) > 2:
            new = list(s)
            new[len(new) - 1] = ''
            s = ''.join(new)
        return s

    def __clean_text(self, text,
                     use_lemmas=True,
                     use_stops=True,
                     use_stemmer=False,
                     max_sents=40,
                     max_words=400):
        sents = sent_tokenize(str(text).lower())
        if len(sents) > max_sents:
            sents = sents[:max_sents]
        full_text = " ".join(sents)
        full_text = str(full_text).replace("\n", " ")

        words = self.tokenizer_words.tokenize(full_text)

        words = [y for x in words for y in ([x] if len(x.split("-")) == 1 else x.split("-"))]
        words = [y for x in words for y in ([x] if len(x.split("|")) == 1 else x.split("|"))]
        words = [y for x in words for y in ([x] if len(x.split("+")) == 1 else x.split("+"))]
        words = [self.__clean_word(x) for x in words]
        words = ["TIME" if self.__is_time(x) else x for x in words]
        words = ["URL" if self.__is_url_str(x) else x for x in words]
        words = ["EMAIL" if self.__is_email(x) else x for x in words]

        if use_lemmas:
            words = [self.lemma.lemmatize(x) for x in words]

        if use_stops:
            words = [i for i in words if i not in self.stop]

        if use_stemmer:
            words = [self.stemmer.stem(x) for x in words]

        words = [i for i in words if i not in "\".,;:-=!@#$%^&*()'<>[]{}™“”\s/?."]
        words = [i for i in words if i.strip() != ""]

        if len(words) > max_words:
            words = words[:max_words]

        return words

    def __call__(self, *args, **kwargs):
        return self.__clean_text(*args, **kwargs)


if __name__ == "__main__":
    text = 'Bukit Merah HDB branch closes temporarily after second employee tests positive for COVID-19 ' \
           'https://t.co/yaMECYuNIN https://t.co/kz7BNBzjez '
    tokenizer = SimpleTokenizer()
    print(tokenizer(text=text, use_stemmer=False))
