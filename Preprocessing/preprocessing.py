from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv
import sys
from math import *

def preproc(border_chi = 1):
    PREPROC_CSV_PATH = "D:\\Uchoba\\Kocheshkov\\Curs\\Data\\data1.csv"
    TRAIN_CSV_PATH = "D:\\Uchoba\\Kocheshkov\\Curs\\Data\\data_train.csv"
    TEST_CSV_PATH = "D:\\Uchoba\\Kocheshkov\\Curs\\Data\\data_test.csv"
    Chi_square = {}
    DF = {}
    CLASS = {}
    COUNT_DOC = 0
    DOCUMENTS = []
    # Структуры для очистки текстов и разделения на токены
    stop_words = set(stopwords.words('russian')+["очень"])
    non_words = ['-', '.', ',', "'", '"', "!"]
    end_words3 = ["ого", "его", "ому", "ему", "ыми", "ими"]
    end_words2 = ["ой", "ый", "ий", "ая", "яя", 
                    "ое", "ее", "ые", "ие", "ую", 
                    "юю", "им", "ым", "ом", "ем",
                    "ых", "их", "ой", "ей", "ла", "ло", "ли"]
    end_words1 = ["а", "я", "ы", "и", "е", "у", "ю", "о"]
    # Очистка данных. Первичная.
    file_ = open(PREPROC_CSV_PATH, "r")#, encoding="utf-8") 
    reader = csv.reader(file_, delimiter = '#')
    count = 0
    for text in reader:
        count += 1
        if len(text) != 2:
            print(count, ':', len(text))
            continue
        COUNT_DOC += 1
        _class, text = text
        if _class in CLASS:
            CLASS[_class] += 1
        else:
            CLASS[_class] = 1
        words = word_tokenize(text)
        tokens = []
        for w in words:
            w = w.lower()
            if w not in stop_words and w not in non_words and len(w)>2:
                if w[-3:] in end_words3:
                    tokens.append(w[:-3])
                    continue
                if w[-2:] in end_words2:
                    tokens.append(w[:-2])
                    continue
                if w[-1:] in end_words1:
                    tokens.append(w[:-1])
                    continue
                tokens.append(w)
        DOCUMENTS.append((_class, tokens))
        for token in set(tokens):
            if token in DF:
                if _class in DF[token]:
                    DF[token][_class] += 1
                else:
                    DF[token][_class] = 1
            else:
                DF[token] = {_class : 1}
    file_.close()
    #Если это были тестовые выборки - то сохраняем в файл и выходим
    if _class == "test":
        data = open(TEST_CSV_PATH, "w") 
        writer = csv.writer(data, delimiter = ',')
        for doc in DOCUMENTS:
            _, doc = doc 
            writer.writerow(doc)
        data.close()
        exit(0)
    #print("Count doc:", COUNT_DOC)
    # Вычисление хи-квадрат критерия
    for _class in CLASS.keys():
        for token in DF.keys():
            if _class in DF[token]:
                A = DF[token][_class]
                C = CLASS[_class] - A
            else:
                A = 0
                C = CLASS[_class]
            B = 0
            for non_cl in DF[token]:
                if non_cl != _class:
                    B += DF[token][non_cl]
            D = (COUNT_DOC - CLASS[_class]) - B
            Z = (A+C)*(B+D)*(A+B)*(C+D)
            if Z != 0:
                Chi_square[(token,_class)] = (COUNT_DOC*(A*D-C*B)**2)/Z 
            else:
                Chi_square[(token,_class)] = 0
            # ниже взаимная информация
            #Z = (COUNT_DOC*A)/((A+C)*(A+B))
            #Chi_square[(token,_class)] = log2(Z) if Z != 0 else 0
            #информационная выгода
            #ZA = (COUNT_DOC*A)/((A+C)*(A+B))
            #ZA = log2(ZA) if ZA != 0 else 0
            #ZB = (COUNT_DOC*B)/((A+B)*(D+B))
            #ZB = log2(ZB) if ZB != 0 else 0
            #ZC = (COUNT_DOC*C)/((C+D)*(A+C))
            #ZC = log2(ZC) if ZC != 0 else 0
            #ZD = (COUNT_DOC*D)/((D+C)*(D+B))
            #ZD = log2(ZD) if ZD != 0 else 0
            #Chi_square[(token,_class)] =(A*ZA/COUNT_DOC) + (B*ZB/COUNT_DOC) + (C*ZC/COUNT_DOC) + (D*ZD/COUNT_DOC)
            #Chi_square[(token,_class)] =A
    # Уменьшение размерности
    F = []
    for doc in DOCUMENTS:
        W = []
        _class, doc = doc
        for token in doc:
            if Chi_square[(token, _class)] > border_chi:#*(max(Chi_square.values())):
                W.append(token)
        F.append((_class, W))
    #Сохраняем в файл наши документы.
    train = open(TRAIN_CSV_PATH, "w") 
    writer = csv.writer(train, delimiter = ',')
    while F != []:
        for c in CLASS.keys():
            for _class, doc in F:
                if c==_class:
                    writer.writerow([_class]+doc)
                    F.remove((_class, doc))
                    break
    train.close()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        preproc()
    elif len(sys.argv) == 2:
        preproc(float(sys.argv[1]))
    else:
        print("Parametrs error!")
        sys.exit(1)