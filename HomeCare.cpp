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

      this->patients.resize(maxKey);

      for (auto& [key, value] : data["patients"].items()) {
        int index = stoi(key) - 1; // Dataset is 1-indexed, which is a pain to work with
        this->patients[index] = Patient(
            value["demand"],
            value["start_time"],
            value["end_time"],
            value["care_time"]
            );
      }

      for (auto p : this->patients) {
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
};
