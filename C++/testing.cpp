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
#define TEST_CSV_PATH "D:\\Uchoba\\Kocheshkov\\Curs\\Data\\test_time.csv"
#define ROCCHIO_CSV_PATH "D:\\Uchoba\\Kocheshkov\\Curs\\Data\\model_rocchio.csv"
using namespace std;

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
struct Document
{
    public:
    vector<string> tokens;
    string _class;
};

int main(int argc, char* argv[])
{
    // на какой выборке тестируемся?
    size_t test_doc;
    if (argc > 1) test_doc = atoi(argv[1]);
    else test_doc = -1;
    // основные переменные
    vector<string> CLASSES;
    vector<Centroids> CENTR;
    vector<Tokens> TOKENS;
    size_t count_doc, iter;
    ifstream csvFile_model,csvFile_test;
    string file_model = ROCCHIO_CSV_PATH;
    string file_test = TEST_CSV_PATH;
    string line, str;
    //Читаем модель
    csvFile_model.open(file_model.c_str());
    if (!csvFile_model.is_open())
    {
        cout << "Error in PATH input file with model!" << endl;
        exit(1);
    }
    while(getline(csvFile_model, line))
    {
        if (line.empty()) continue;
        line.pop_back();line.pop_back();
        stringstream lineStream(line);
        getline(lineStream, str, ',');
        count_doc = atoi(str.c_str());
        while(getline(lineStream, str, ','))
        {
            Tokens _token;
            _token.token = str;
            TOKENS.push_back(_token);
        }
        break;
    }
    while(getline(csvFile_model, line))
    {
        if (line.empty()) continue;
        line.pop_back();line.pop_back();
        stringstream lineStream(line);
        iter = 0;
        while(getline(lineStream, str, ','))
        {
            TOKENS[iter++].count = atoi(str.c_str());
        }
        break;
    }
    while(getline(csvFile_model, line))
    {
        if (line.empty()) continue;
        line.pop_back();line.pop_back();
        stringstream lineStream(line);
        getline(lineStream, str, ',');
        Centroids _centr;
        _centr._class = str;
        CLASSES.push_back(str);
        while(getline(lineStream, str, ','))
        {
            _centr.centr.push_back(atof(str.c_str()));
        }
        CENTR.push_back(_centr);
    }
    csvFile_model.close();
    //Читаем тестовую выборку.
    csvFile_test.open(file_test.c_str());
    if (!csvFile_test.is_open())
    {
        cout << "Error in PATH input file with test-data!" << endl;
        exit(1);
    }
    while(getline(csvFile_test, line))
    {
        if (line.empty()) continue;
        if (test_doc-- == 0) break;
        line.pop_back();
        vector<double> location_vector_test;
        vector<double> vector_of_weights;
        Document *d = new Document;
        d->_class = "";
        stringstream lineStream(line);
        while(getline(lineStream, str, ','))
        {
            d->tokens.push_back(str);
        }
        for(size_t num = 0; num < TOKENS.size(); num++)
        {
            size_t tf = 0;
            for(size_t tok = 0; tok < d->tokens.size(); tok++)
            {
                if((d->tokens[tok]) == TOKENS[num].token) tf++;
            }
            double w = tf * log2(count_doc / TOKENS[num].count); 
            vector_of_weights.push_back(w);
        }
        double llwll = 0.0;
        for(size_t weight = 0; weight < vector_of_weights.size(); weight++)
        {
            llwll += vector_of_weights[weight]*vector_of_weights[weight];
        }
        llwll = sqrt(llwll);
        for(size_t weight = 0; weight < vector_of_weights.size(); weight++)
        {
            location_vector_test.push_back((llwll != 0)?(vector_of_weights[weight]/llwll):0.0);
        }
        string _class_;
        double distance = 1000000;
        for(size_t cls = 0; cls < CLASSES.size(); cls++)
        {
            double res = 0, index;
            for(size_t _centr = 0; _centr < CENTR.size(); _centr++)
            {
                if (CENTR[_centr]._class == CLASSES[cls])
                {
                    index = _centr;
                    break;
                }
            }
            for(size_t i=0; i < location_vector_test.size(); i++)
            {
                res += pow((CENTR[index].centr[i] - location_vector_test[i]),2.0);
            }
            res = sqrt(res);
            if (res <= distance)
            {
                _class_ = CLASSES[cls];
                distance = res;
            }
        }
        //cout << _class_ << endl;
    }
    csvFile_test.close();
}