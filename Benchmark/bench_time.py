import subprocess
import matplotlib.pyplot as plt
import time
# Основные переменные
train_count = [50,100,500,1000,3000,7000,10000,30000,50000,80000,100000]
time_train_global_first = []
time_train_global_second = []
test_count = [1,10,100,500,1000,3000,5000,7000,10000,15000,20000]
time_test_global_first = []
time_test_global_second = []
ROCCHIO_PATH = ["D:\\Uchoba\\Kocheshkov\\Curs\\Python_0\\",
                "D:\\Uchoba\\Kocheshkov\\Curs\\Python_1\\",
                "D:\\Uchoba\\Kocheshkov\\Curs\\C++_1\\",
                "D:\\Uchoba\\Kocheshkov\\Curs\\C++_1\\O1\\",
                "D:\\Uchoba\\Kocheshkov\\Curs\\C++_1\\O2\\",
                "D:\\Uchoba\\Kocheshkov\\Curs\\C++_1\\O3\\",
                "D:\\Uchoba\\Kocheshkov\\Curs\\C++_1\\march\\",
                "D:\\Uchoba\\Kocheshkov\\Curs\\C++_1\\O2\\march\\",
                "D:\\Uchoba\\Kocheshkov\\Curs\\C++_2\\",
                "D:\\Uchoba\\Kocheshkov\\Curs\\C++_2\\O2\\",
                "D:\\Uchoba\\Kocheshkov\\Curs\\C++_3\\"]
# Небольшой пользовательский ввод
while True:
    num1 = int(input("First\n"+''.join(["{}. {}\n".format(i, ROCCHIO_PATH[i]) for i in range(len(ROCCHIO_PATH))])+"Enter: "))
    if -1 > num1 >= len(ROCCHIO_PATH):
        print("Error input.")
        continue
    num2 = int(input("Second\n"+''.join(["{}. {}\n".format(i, ROCCHIO_PATH[i]) for i in range(len(ROCCHIO_PATH))])+"Enter: "))
    if -1 > num2 >= len(ROCCHIO_PATH) or num1 == num2:
        print("Error input.")
        continue
    break

# Эксперименты
def run_module(num):
    flag = num < 2
    return_list_train = []
    for count in train_count:
        print(count)
        list_save_train = []
        for i in range(10):
            cmd = 'python '+ROCCHIO_PATH[num]+'training.py '+str(count) if flag else ROCCHIO_PATH[num]+'training.exe '+str(count)
            start, go = time.time(), subprocess.Popen(cmd, shell=True)
            go.wait()
            stop = time.time()
            list_save_train.append(stop-start)
        list_save_train.remove(max(list_save_train))
        list_save_train.remove(min(list_save_train))
        return_list_train.append(sum(list_save_train)/len(list_save_train))
    return_list_test = []
    for count in test_count:
        print(count)
        list_save_test = []
        for i in range(10):
            cmd = 'python '+ROCCHIO_PATH[num]+'testing.py '+str(count) if flag else ROCCHIO_PATH[num]+'testing.exe '+str(count)
            start, go = time.time(), subprocess.Popen(cmd, shell=True)
            go.wait()
            stop = time.time()
            list_save_test.append(stop-start)
        list_save_test.remove(max(list_save_test))
        list_save_test.remove(min(list_save_test))
        return_list_test.append(sum(list_save_test)/len(list_save_test))
    return return_list_train,return_list_test

time_train_global_first, time_test_global_first = run_module(num1)
time_train_global_second, time_test_global_second = run_module(num2)

# Построение графика по тренировке
LIM_MAX = max(time_test_global_second+time_train_global_first+time_train_global_second+time_test_global_first)

def save_graphics(list1, list2, n1, n2, _type = True):
    fig, ax = plt.subplots()
    ax.set_title('Скорость обучения модели' if _type else 'Скорость классификации модели')
    ax.set_xlabel('Количество документов')
    ax.set_ylabel('Время выполнения, сек')
    ax.grid()
    ax.plot(train_count if _type else test_count, list1, 
            label=ROCCHIO_PATH[n1][26:]+('training' if _type else 'testing'))
    ax.plot(train_count if _type else test_count, list2, 
            label=ROCCHIO_PATH[n2][26:]+('training' if _type else 'testing'))
    ax.legend()
    ax.set(ylim=(0.0, LIM_MAX*1.1))
    fig.savefig('time_train{}{}'.format(n1,n2) if _type else 'time_test{}{}'.format(n1,n2))

save_graphics(time_train_global_first, time_train_global_second, num1, num2, True)
save_graphics(time_test_global_first, time_test_global_second, num1, num2, False)
