import subprocess
import csv
import sys
import matplotlib.pyplot as plt

def metrics(answer_list, prediction_list):
    accuracy = 0
    local_prec = []
    local_recall = []
    local_f = []
    for _class in set(answer_list):
        TP = TN = FP = FN = 0
        for pred in range(len(answer_list)):
            if _class == answer_list[pred] == prediction_list[pred]:
                TP += 1
                accuracy += 1
            elif _class == answer_list[pred] != prediction_list[pred]:
                FN += 1
            elif _class == prediction_list[pred] != answer_list[pred]:
                FP += 1
            else:
                TN += 1
        local_prec.append(TP/(TP+FP) if TP+FP!= 0 else 0)
        local_recall.append(TP/(TP+FN)if TP+FN!= 0 else 0)
        if local_recall[-1]+local_prec[-1] != 0:
            _f = 2*local_recall[-1]*local_prec[-1]/(local_recall[-1]+local_prec[-1])
        else:
            _f = 0
        local_f.append(_f)
    precision = sum(local_prec)/len(local_prec)
    accuracy = accuracy/len(answer_list)
    recall = sum(local_recall)/len(local_recall)
    f = sum(local_f)/len(local_f)
    return accuracy, precision, recall, f

ROCCHIO_TRAINING_PATH = "D:\\Uchoba\\Kocheshkov\\Curs\\Python_0\\training.py"
ROCCHIO_TESTING_PATH = "D:\\Uchoba\\Kocheshkov\\Curs\\Python_0\\testing.py"
NAIVE_TESTING_PATH = "D:\\Uchoba\\Kocheshkov\\Curs\\NaiveBayesian\\NBtesting.py"
NAIVE_TRAINING_PATH = "D:\\Uchoba\\Kocheshkov\\Curs\\NaiveBayesian\\NBtraining.py"
ANSWER_PATH = "D:\\Uchoba\\Kocheshkov\\Curs\\Data\\answer.csv"

ACCURACY = []
PRECISION = []
RECALL = []
F = []

answer_file = open(ANSWER_PATH, "r")
answer = []
reader = csv.reader(answer_file)
for doc in reader:
    if doc == []:
        continue
    answer.append(doc[0])
answer_file.close()
print(answer)

for count_train in range(6, 275, 6):
    #Обучаем модели
    go = subprocess.Popen(
               'python '+NAIVE_TRAINING_PATH+' '+str(count_train)+' '+'p',
               stdout=subprocess.PIPE,
               stderr=subprocess.STDOUT,
               shell=True)
    go.wait()
    go = subprocess.Popen(
               'python '+ROCCHIO_TRAINING_PATH+' '+str(count_train)+' '+'p',
               stdout=subprocess.PIPE,
               stderr=subprocess.STDOUT,
               shell=True)
    go.wait()
    local_rocchio_class = []
    local_naive_class = []
    #Используем модели
    go = subprocess.Popen(
                'python '+ROCCHIO_TESTING_PATH,
                stdout=subprocess.PIPE,
                universal_newlines=True,
                stderr=subprocess.STDOUT,
                text=True,encoding="utf-8")
    while go.poll() is None:
        try:
            out, err = go.communicate()
            local_rocchio_class += out
        except:
            continue
    local_rocchio_class = ''.join(local_rocchio_class).split('\n')[:-1]
    go = subprocess.Popen(
                'python '+NAIVE_TESTING_PATH,
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,encoding="utf-8")
    while go.poll() is None:
        try:
            out, err = go.communicate()
            local_naive_class += out
        except:
            continue
    local_naive_class = ''.join(local_naive_class).split('\n')[:-1]
    #Считаем характеристики локально для класса и глобально добавляем в глобальный список
    a,p,r,f = metrics(answer, local_rocchio_class)
    ac,pr,re,f_ = metrics(answer, local_naive_class)
    ACCURACY.append((a,ac))
    PRECISION.append((p,pr))
    RECALL.append((r,re))
    F.append((f,f_))
#Рисуем фигуры
print(ACCURACY)
print(PRECISION)
print(RECALL)
print(F)

fig, (pre, acc) = plt.subplots(nrows=2, ncols=2, figsize = (10,10))
pre, rec = pre[0], pre[1]
acc, _f = acc[0], acc[1]
acc.set_title('Метрика: Доля правильных ответов')
pre.set_title('Метрика: Точность')
rec.set_title('Метрика: Полнота')
_f.set_title('Метрика: F-мера')
acc.set_xlabel('Количество документов')
pre.set_xlabel('Количество документов')
rec.set_xlabel('Количество документов')
_f.set_xlabel('Количество документов')
acc.set_ylabel('accuracy')
pre.set_ylabel('precision')
rec.set_ylabel('recall')
_f.set_ylabel('F-мера')
acc.grid()      # включение отображение сетки
pre.grid()
rec.grid()
_f.grid()
acc.plot(range(6, 275, 6), [i[0] for i in ACCURACY], label="Алгоритм Роккио")
acc.plot(range(6, 275, 6), [i[1] for i in ACCURACY], label="Наивный Байессовский")  # построение графика
pre.plot(range(6, 275, 6), [i[0] for i in PRECISION], label="Алгоритм Роккио")
pre.plot(range(6, 275, 6), [i[1] for i in PRECISION], label="Наивный Байессовский")
rec.plot(range(6, 275, 6), [i[0] for i in RECALL], label="Алгоритм Роккио")
rec.plot(range(6, 275, 6), [i[1] for i in RECALL], label="Наивный Байессовский")
_f.plot(range(6, 275, 6), [i[0] for i in F], label="Алгоритм Роккио")
_f.plot(range(6, 275, 6), [i[1] for i in F], label="Наивный Байессовский")
acc.legend()
pre.legend()
rec.legend()
_f.legend()
fig.savefig('metrics')