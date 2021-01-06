import math as m
import csv
import time
import sys
import numpy as np

def training_rocchio(count_doc_ = None):
    TRAIN_CSV_PATH ='D:\\Uchoba\\Kocheshkov\\Curs\\Data\\data_train.csv'
    ROCCHIO_CSV_PATH = "D:\\Uchoba\\Kocheshkov\\Curs\\Data\\model_rocchio.csv"
    TEST_TIME_CSV_PATH = "D:\\Uchoba\\Kocheshkov\\Curs\\Data\\test_time.csv"
    TOKENS = {}
    CLASSES = []
    COUNT_DOCUMENTS = 0
    CENTROIDS = []
    DOCUMENTS = []
    # открыть либо иск. данные либо настоящие
    if count_doc_ is None:
        train_file = open(TRAIN_CSV_PATH, "r")
    else:
        train_file = open(TEST_TIME_CSV_PATH, "r")
    reader = csv.reader(train_file, delimiter = ',')
    # прочитать тренировочную выборку
    for document in reader:
        if len(document)<2:
            continue
        if count_doc_ == 0:
            break
        elif count_doc_ is not None:
            count_doc_ -= 1
        COUNT_DOCUMENTS += 1
        DOCUMENTS.append(document)
        classes, tokens = document[0], document[1:]
        if not (classes in CLASSES):
            CLASSES.append(classes)
        tokens = set(tokens)
        for token in tokens:
            TOKENS[token] = TOKENS[token] + 1 if token in TOKENS else 1
    train_file.close()
    # основные вычисления
    for _class in CLASSES:
        sum_vector_class = np.zeros((len(TOKENS)), float) 
        num_doc_class = 0
        for document in DOCUMENTS:
            classes, tokens = document[0], document[1:]
            if classes == _class:
                num_doc_class += 1
                vector_of_weights = np.zeros((len(TOKENS)), float)
                tok_array = np.array(list(TOKENS.values()), int)
                tf = np.zeros((len(TOKENS)), int)
                ii = 0
                for token_base in TOKENS.keys():
                    tf[ii] += tokens.count(token_base)
                    ii += 1
                vector_of_weights = tf * np.log2(COUNT_DOCUMENTS / tok_array)
                vector_of_weights_sqr = vector_of_weights*vector_of_weights
                llwll = m.sqrt(vector_of_weights_sqr.sum())
                sum_vector_class = vector_of_weights / llwll
        sum_vector_class = sum_vector_class / num_doc_class
        CENTROIDS.append(([_class] + sum_vector_class.tolist()))
    # запись модели
    model = open(ROCCHIO_CSV_PATH, "w") 
    writer = csv.writer(model, delimiter = ',')
    writer.writerow([COUNT_DOCUMENTS]+list(TOKENS.keys()))
    writer.writerow(list(TOKENS.values()))
    for centr_classes in CENTROIDS:
        writer.writerow(centr_classes)
    model.close()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        training_rocchio()
    elif len(sys.argv) == 2:
        training_rocchio(int(sys.argv[1]))
    else:
        print("Parametrs error!")
        sys.exit(1)