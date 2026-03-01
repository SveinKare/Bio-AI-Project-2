#include <iomanip>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include "Patient.cpp"

using namespace std;
using json = nlohmann::json;

class HomeCare {
  private:
    int numberOfNurses;
    int capacity;
    double benchmark;
    int returnTime;
    int nbrPatients;
    vector<Patient> patients;
    vector<vector<double>> travelTimes;

    bool readDataset(string path) {
      ifstream file(path);

      if (!file.is_open()) {
        cerr << "Could not open dataset file" << endl;
        return false;
      }

      json data = json::parse(file);

      this->numberOfNurses = data["nbr_nurses"];
      this->capacity = data["capacity_nurse"];
      this->benchmark = data["benchmark"];
      this->returnTime = data["depot"]["return_time"];

      // Instantiate patients here
      int maxKey = 0;
      for (auto& [key, value] : data["patients"].items()) {
        int index = stoi(key);
        if (index > maxKey) maxKey = index;
      }

      this->nbrPatients = maxKey;
      this->patients.resize(maxKey + 1);

      for (auto& [key, value] : data["patients"].items()) {
        int index = stoi(key);
        this->patients[index] = Patient(
            value["demand"],
            value["start_time"],
            value["end_time"],
            value["care_time"]
            );
      }

      for (int i = 1; i <= nbrPatients; ++i) {
        if (!patients[i].valid()) {
          cout << "Invalid patient " << i << ": " << patients[i].toString() << endl;
        }
      }

      this->travelTimes = data["travel_times"].get<vector<vector<double>>>();

      return true;
    }


  public:
    HomeCare() {

    }

    void init(string datasetPath) {
      readDataset(datasetPath);
    }

    void run() {

    }

    int getNumberOfNurses() const { return numberOfNurses; }
    int getCapacity() const { return capacity; }
    double getBenchmark() const { return benchmark; }
    int getReturnTime() const { return returnTime; }
    const vector<Patient>& getPatients() const { return patients; }
    const vector<vector<double>>& getTravelTimes() const { return travelTimes; }
    int getNbrPatients() const { return nbrPatients; }

    double getTravelTime(int from, int to) const {
      size_t i = static_cast<size_t>(from);
      size_t j = static_cast<size_t>(to);
      if (i >= travelTimes.size() || j >= travelTimes[0].size()) return 0;
      return travelTimes[i][j];
    }
};
