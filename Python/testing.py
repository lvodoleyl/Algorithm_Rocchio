import math as m
import csv
import sys
import numpy as np

def testing_rocchio(count_doc_ = None):
    COUNT_DOCUMENTS = 0
    TEST_CSV_PATH = "D:\\Uchoba\\Kocheshkov\\Curs\\Data\\data_test.csv"
    ROCCHIO_CSV_PATH = "D:\\Uchoba\\Kocheshkov\\Curs\\Data\\model_rocchio.csv"
    TEST_TIME_CSV_PATH = "D:\\Uchoba\\Kocheshkov\\Curs\\Data\\test_time.csv"
    TOKENS = []
    CENTROIDS = []
    CLASSES = []
    DF = []
    #to_file = ""
    
    model = open(ROCCHIO_CSV_PATH, "r") 
    reader = csv.reader(model)
    while True:
        TOKENS = next(reader)
        if TOKENS != []:
            break
    COUNT_DOCUMENTS, TOKENS = int(TOKENS[0]), TOKENS[1:]
    while True:
        DF = next(reader)
        if DF != []:
            break
    for _class in reader:
        if _class == []:
            continue
        CENTROIDS.append(_class)
        classes = _class[0]
        CLASSES.append(classes)
    model.close()

    if count_doc_ is None:
        test_file = open(TEST_CSV_PATH, "r")
    else:
        test_file = open(TEST_TIME_CSV_PATH, "r")
    reader = csv.reader(test_file)

    for test in reader:
        if test ==[]:
            continue
        if count_doc_ == 0:
            break
        elif count_doc_ is not None:
            count_doc_ -= 1
        test = list(test)
        llenl = len(TOKENS)
        location_vector_test = np.zeros(llenl, float)
        vector_of_weights=np.zeros(llenl, float)
        tf = np.zeros((len(TOKENS)), int)
        DF_array = np.array(DF,int)
        for num in range(llenl):
            tf[num] += test.count(TOKENS[num])
        vector_of_weights = tf * np.log2(COUNT_DOCUMENTS / DF_array)
        vector_of_weights_sqr = vector_of_weights * vector_of_weights
        llwll = m.sqrt(vector_of_weights_sqr.sum())
        location_vector_test = vector_of_weights_sqr / llwll
        np.putmask(location_vector_test, location_vector_test == float('inf'), 0.0)
        c, distance = "", float('inf')
        for _class in CLASSES:
            res = 0
            for _centr in CENTROIDS:
                classes, centr = _centr[0], _centr[1:]
                if classes == _class:
                    centr = np.array(centr, float)
                    break
            res = m.sqrt(np.power((centr - location_vector_test), 2).sum())
            if res <= distance:
                c, distance = _class, res
    #    to_file += " "+c
    #with open("D:\\Uchoba\\Kocheshkov\\Curs\\benchmark_two_class\\res.txt", 'w') as res_f:
    #     res_f.write(to_file[1:])
    test_file.close()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        testing_rocchio()
    elif len(sys.argv) == 2:
        testing_rocchio(int(sys.argv[1]))
    else:
        print("Parametrs error!")
        sys.exit(1)