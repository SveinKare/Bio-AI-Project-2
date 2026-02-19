#pragma once

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "Patient.hpp"

using json = nlohmann::json;

class HomeCare {
private:
    int numberOfNurses;
    int capacity;
    double benchmark;
    int returnTime;
    std::vector<Patient> patients;
    std::vector<std::vector<double>> travelTimes;

    bool readDataset(std::string path);

public:
    HomeCare();

    void init(std::string datasetPath);
    void run();

    int getNumberOfNurses() const;
    int getCapacity() const;
    double getBenchmark() const;
    int getReturnTime() const;
    const std::vector<Patient>& getPatients() const;
    const std::vector<std::vector<double>>& getTravelTimes() const;
    int getNbrPatients() const;
    double getTravelTime(int from, int to) const;
};
