#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <iterator>
#include <cmath>
#include <math.h> 

//#define TRAIN_CSV_PATH "D:\\Uchoba\\Kocheshkov\\Curs\\Data\\data_train.csv"
#define TRAIN_CSV_PATH "D:\\Uchoba\\Kocheshkov\\Curs\\Data\\test_time.csv"
#define ROCCHIO_CSV_PATH "D:\\Uchoba\\Kocheshkov\\Curs\\Data\\model_rocchio.csv"
using namespace std;

struct Document
{
    public:
    vector<string> tokens;
    string _class;
};
struct Tokens
{
    public:
    string token;
    int count;
};
struct Centroids
{
    public:
    string _class;
    vector<double> centr;
};


int main(int argc, char* argv[])
{
    // на какой выборке тестируемся?
    size_t train_doc;
    if (argc > 1) train_doc = atoi(argv[1]);
    else train_doc = -1; 
    vector<Document> DOCS;
    vector<string> CLASSES;
    vector<Centroids> CENTR;
    vector<Tokens> TOKENS;
    bool flag;
    size_t count_doc;
    //vector<Centroids> result;
    ifstream csvFile;
    string file_name = TRAIN_CSV_PATH;
    string line, str;
    //Читаем файл, и считываем документы
    csvFile.open(file_name.c_str());
    if (!csvFile.is_open())
    {
        cout << "Error in PATH input file!" << endl;
        exit(1);
    }
    while(getline(csvFile, line))
    {
        //считываем один документ
        if (line.empty()) continue;
        if (train_doc-- == 0) break;
        line.pop_back();
        count_doc++;
        Document *d = new Document;
        stringstream lineStream(line);
        getline(lineStream,d->_class,',');
        while (getline(lineStream, str, ','))
        {
            d->tokens.push_back(str);
            //Считаем df 
            flag = true;
            for(size_t i = 0; i < d->tokens.size()-1; i++)
            {
                flag = flag && (d->tokens[i] != str);
            }
            for(size_t i = 0; i < TOKENS.size(); i++)
            {
                if ((TOKENS[i].token == str)&&(flag))
                {
                    TOKENS[i].count++;
                    flag = false;
                    break;
                }
            }
            if (flag) 
            {
                Tokens tok;
                tok.token = str;
                tok.count = 1;
                TOKENS.push_back(tok);
            }            
        }
        //есть ли там наш класс?
        flag = true;
        for(size_t i = 0; i < CLASSES.size(); i++)
        {
            if (CLASSES[i] == d->_class) flag = false;
        }
        if (flag) CLASSES.push_back(d->_class);
        DOCS.push_back(*d);
    }
    csvFile.close();
    //Основные вычисления
    for (size_t _class = 0; _class < CLASSES.size(); _class++)
    {
        vector<double> sum_vector_class(TOKENS.size());
        size_t num_doc_class = 0;
        for (size_t _doc = 0; _doc < DOCS.size(); _doc++)
        {
            if (DOCS[_doc]._class == CLASSES[_class])
            {
                num_doc_class++;
                vector<double> vector_of_weights;
                for(size_t tok = 0; tok < TOKENS.size(); tok++)
                {
                    size_t tf = 0;
                    for(size_t i = 0; i < DOCS[_doc].tokens.size(); i++)
                    {
                        if (DOCS[_doc].tokens[i] ==TOKENS[tok].token) tf++; 
                    }
                    double w = tf * log2(count_doc / TOKENS[tok].count);
                    vector_of_weights.push_back(w);
                }
                double llwll = 0;
                for(size_t weight = 0; weight < vector_of_weights.size();weight++)
                {
                    llwll += vector_of_weights[weight]*vector_of_weights[weight];
                }
                llwll = sqrt(llwll);
                for(size_t weight = 0; weight < vector_of_weights.size();weight++)
                {
                    sum_vector_class[weight] += vector_of_weights[weight] / llwll;
                }
            }
        }
        for(size_t i = 0; i < sum_vector_class.size();i++)
        {
            sum_vector_class[i] = sum_vector_class[i] / num_doc_class;
        }
        Centroids _centr;
        _centr._class = CLASSES[_class];
        _centr.centr = sum_vector_class;
        CENTR.push_back(_centr);
    }
    //Осталось записать!
    fstream csvFile_model;
    string file_model = ROCCHIO_CSV_PATH;
    csvFile_model.open(file_model.c_str(), ios::out);
    if (!csvFile_model.is_open())
    {
        ofstream ofs(ROCCHIO_CSV_PATH);
        csvFile.open(file_model.c_str());
    }
    csvFile_model << count_doc << ',';
    for(size_t _tok = 0; _tok < TOKENS.size(); _tok++)
        csvFile_model << TOKENS[_tok].token << ',';
    csvFile_model << endl;
    for(size_t _tok = 0; _tok < TOKENS.size(); _tok++)
        csvFile_model << TOKENS[_tok].count << ',';
    csvFile_model << endl;
    for(size_t _c_ = 0; _c_ < CENTR.size(); _c_++)
    {
        csvFile_model << CENTR[_c_]._class << ',';
        for(size_t val = 0; val < CENTR[_c_].centr.size(); val++)
            csvFile_model << CENTR[_c_].centr[val] << ',';
        csvFile_model << endl;
    }
    csvFile_model.close();
}