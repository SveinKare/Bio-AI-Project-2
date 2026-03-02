#include "HomeCare.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;

HomeCare::HomeCare() {}

void HomeCare::init(std::string datasetPath) {
    readDataset(datasetPath);
}

void HomeCare::testInit() {
  this->numberOfNurses = 3;
  this->patients.resize(6);
  this->travelTimes.resize(6);
  for (size_t i = 0; i < 6; i++) {
    this->travelTimes[i].resize(6);
    for (size_t j = 0; j < 6; j++) {
      this->travelTimes[i][j] = (i-j) * (i-j);
    }
  }

}

bool HomeCare::readDataset(std::string path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Could not open dataset file" << std::endl;
        return false;
    }
    json data = json::parse(file);
    this->numberOfNurses = data["nbr_nurses"];
    this->capacity = data["capacity_nurse"];
    this->benchmark = data["benchmark"];
    this->returnTime = data["depot"]["return_time"];

    int maxKey = 0;
    for (auto& [key, value] : data["patients"].items()) {
        int index = stoi(key);
        if (index > maxKey) maxKey = index;
    }
    this->patients.resize(maxKey + 1);
    this->patients[0] = Patient();
    for (auto& [key, value] : data["patients"].items()) {
        int index = stoi(key);
        this->patients[index] = Patient(
            value["demand"],
            value["start_time"],
            value["end_time"],
            value["care_time"],
            value["x_coord"],
            value["y_coord"]
        );
    }
    bool first = true;
    for (auto p : this->patients) {
        if (first) { first = false; continue; }
        if (!p.valid())
            std::cout << "Invalid patient: " << p.toString() << std::endl;
    }
    this->travelTimes = data["travel_times"].get<std::vector<std::vector<double>>>();
    return true;
}

int HomeCare::getNumberOfNurses() const { return numberOfNurses; }
int HomeCare::getCapacity() const { return capacity; }
double HomeCare::getBenchmark() const { return benchmark; }
int HomeCare::getReturnTime() const { return returnTime; }
const std::vector<Patient>& HomeCare::getPatients() const { return patients; }
const std::vector<std::vector<double>>& HomeCare::getTravelTimes() const { return travelTimes; }
int HomeCare::getNbrPatients() const { return static_cast<int>(patients.size() - 1); }

double HomeCare::getTravelTime(int from, int to) const {
    size_t i = (from < 0) ? 0 : static_cast<size_t>(from) + 1;
    size_t j = (to < 0) ? 0 : static_cast<size_t>(to) + 1;
    if (i >= travelTimes.size() || j >= travelTimes[0].size()) return 0;
    return travelTimes[i][j];
}

bool HomeCare::allPatientsPresent(const vector<int>& solution) const {
  int nPatients = this->getNbrPatients();
  std::vector<bool> present(nPatients + 1, false); 

  for (int patient : solution) {
    if (patient == 0) continue;
    if (patient < 0 || patient > nPatients) {
      return false;  // Invalid patient ID
    }
    if (present[patient]) {
      return false;  // Duplicate found
    }
    present[patient] = true;
  }

  for (int i = 1; i <= nPatients; i++) {
    if (!present[i]) {
      return false;
    }
  }
  return true;
}

pair<double, double> HomeCare::calculateFitness(const vector<int>& gene, double penalty) const {
  if (gene.empty()) {
    return pair<double, double>(INFINITY, INFINITY);
  }
  // The logic in the algorithm requires all patients to be present, so this is considered an error
  if (!allPatientsPresent(gene)) {
    throw runtime_error("Invalid gene found");
  }
  if (gene[0] != 0 || gene[gene.size()-1] != 0) {
    throw runtime_error("Gene is missing 0 at start or end");
  }

  // How much we have violated the constraints
  int excessStrain = 0;
  double timeViolations = 0.0;

  double sum = 0.0;
  double currentRoute = 0.0;
  int currentDemand = 0;
  double time = 0.0;
  for (size_t i = 1; i < gene.size(); i++) {
    int currentPatient = gene[i];
    double travelTime = this->travelTimes[gene[i-1]][currentPatient]; 
    time += travelTime;
    currentRoute += travelTime;

    if (gene[i] == 0) { // Depot
      sum += currentRoute;

      // Constraints
      excessStrain += max(0, currentDemand - this->capacity);
      timeViolations += max(0.0, time - this->returnTime);

      currentRoute = 0.0;
      time = 0.0;
      currentDemand = 0;
      continue;
    }

    // Waiting
    if (time < patients[currentPatient].getStartTime()) {
      time  += patients[currentPatient].getStartTime() - time;
    }

    time += patients[currentPatient].getCareTime();

    // If the nurse is too late, we add a penalty
    if (time > patients[currentPatient].getEndTime()) {
      timeViolations += time - patients[currentPatient].getEndTime();
    }
    currentDemand += patients[currentPatient].getDemand();
  }

  return pair<double, double>(sum, penalty*excessStrain + penalty*timeViolations);
}


