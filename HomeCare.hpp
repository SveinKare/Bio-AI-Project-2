#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "Patient.hpp"

using json = nlohmann::json;
using namespace std;

class HomeCare {
private:
    int numberOfNurses;
    int capacity;
    double benchmark;
    int returnTime;
    vector<Patient> patients;
    vector<vector<double>> travelTimes;

    bool readDataset(string path);

public:
    HomeCare();

    void init(string datasetPath);
    void testInit();

    int getNumberOfNurses() const;
    int getCapacity() const;
    double getBenchmark() const;
    int getReturnTime() const;
    const vector<Patient>& getPatients() const;
    const vector<vector<double>>& getTravelTimes() const;
    int getNbrPatients() const;
    double getTravelTime(int from, int to) const;
    bool allPatientsPresent(const vector<int>& solution) const;
    double calculateFitness(const vector<int>& gene, double penalty) const;
};
