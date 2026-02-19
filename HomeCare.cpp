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

      this->patients.resize(maxKey + 1);
      this->patients[0] = Patient(); // Inserting dummy at the start to get 1-indexed vector

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
        // First element is a dummy
        if (first) {
          first = false;
          continue;
        }
        if (!p.valid()) {
          cout << "Invalid patient: " << p.toString() << endl;
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
    int getNbrPatients() const { return static_cast<int>(patients.size()-1); }

    double getTravelTime(int from, int to) const {
      size_t i = (from < 0) ? 0 : static_cast<size_t>(from) + 1;
      size_t j = (to < 0) ? 0 : static_cast<size_t>(to) + 1;
      if (i >= travelTimes.size() || j >= travelTimes[0].size()) return 0;
      return travelTimes[i][j];
    }
};
