#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stack>
#include <vector>

#include <assert.h>
#include <math.h>
#include <stdio.h>

#define PI 3.141592653589793
#define MINDIST 1.5
#define IN
#define OUT

using namespace std;

struct Item {
    string photoId;
    string userId;
    string dateTaken;
    double longitude;
    double latitude;
};


bool load_records(string& fname, OUT vector<Item>& records)
{
    records.clear();
    ifstream ifs(fname.c_str());
    if (!ifs.is_open()) {
        cerr << "ERROR: open file \"" << fname << "\" failed" << endl;
        return false;
    }

    string line;
    while (getline(ifs, line, '\n')) {
        istringstream iss(line);
        Item item;
        iss >> item.photoId 
            >> item.userId 
            >> item.dateTaken 
            >> item.longitude 
            >> item.latitude;
        /*
        clog << item.photoId    << ", "
             << item.userId     << ", "
             << item.dateTaken  << ", "
             << item.longitude  << ", "
             << item.latitude << endl;
        */
        records.push_back(item);
    }

    ifs.close();
    return true;
}


double calc_dist(double longitude1, double latitude1, double longitude2, double latitude2)
{
    double radius = 6371.009; //mean earth radius is 6371.009km, en.wikipedia.org/wiki/Earth_radius#Mean_radius

    //convert degrees to radians
    double lng1 = 1.0/180.0 * PI * longitude1;
    double lng2 = 1.0/180.0 * PI * longitude2;
    double lat1 = 1.0/180.0 * PI * latitude1;
    double lat2 = 1.0/180.0 * PI * latitude2;
    double dlng = fabs(lng1 - lng2);
    double dlat = fabs(lat1 - lat2);
    double term1 = sin(dlat / 2.0);
    double term2 = sin(dlng / 2.0);

    double dist = 2.0 * radius * asin( sqrt(term1*term1 + cos(lat1) * cos(lat2)* term2 * term2) );
    return dist;
}


void filter_records(IN vector<Item>& records)
{
    float dmin = MINDIST; //km
    vector<bool> indicator(records.size(), true); //true if this record should be keeped
    vector<int> minidx(records.size(), -1); //index of records[i]'s nearest neighbor

    stack<int> S; //contain index of record whose indicator is False but its nearest neighbors unnotified
    for (unsigned int i = 0; i < records.size(); i++) {
        float mindist = dmin + 1;
        if (i % 100 == 0) { clog << i << endl; }
        for (unsigned int j = 0; j < records.size(); j++) {
            if (i == j) continue;
            double dist = calc_dist(records[i].longitude, records[i].latitude, records[j].longitude, records[j].latitude);
            if (dist < mindist) {
                mindist = dist;
                minidx[i] = j;
            }
        }
        if (mindist > dmin) {
            indicator[i] = false;
            S.push(i);
        }
    }

    while (!S.empty()) {
        clog << "#element in stack: " << S.size() << endl;
        int idx = S.top(); S.pop();
        for (unsigned int j = 0; j < records.size(); j++) {
            if (indicator[j] && minidx[j] == idx) { //re-calculate the nearest neighbor for j
                double mindist = dmin + 1;
                for (unsigned int k = 0; k < records.size(); k++) {
                    if (j == k || indicator[k] == false) continue;
                    double dist = calc_dist(records[j].longitude, records[j].latitude, records[k].longitude, records[k].latitude);
                    if (dist < mindist) {
                        mindist = dist;
                        minidx[j] = k;
                    }
                }
                if (mindist > dmin) {
                    indicator[j] = false;
                    S.push(j);
                }
            }
        }
    }

    for (unsigned int i = 0; i < records.size(); i++) {
        if (indicator[i]) {
            printf("%s,%s,%s,%.6f,%.6f\n", records[i].photoId.c_str(), records[i].userId.c_str(), 
                                           records[i].dateTaken.c_str(), records[i].longitude, records[i].latitude);
        }
    }
}


int main(int argc, char** argv)
{
    //printf("%.15f\n", calc_dist(149.128183, -35.2828229, 149.1237519, -35.2768234));
    ///*
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " INPUT_FILE" << endl;
        return 0;
    }

    string fin = argv[1];
    vector<Item> records;
    assert(load_records(fin, records) == true);
    filter_records(records);
    //*/

    return 0;
}
