#include <sstream>
#include <string>

using namespace std;

class Patient {
  private:
    int demand;
    int startTime;
    int endTime;
    int careTime;
  public:
    Patient() : demand(-1), startTime(-1), endTime(-1), careTime(-1) {};

    Patient(int demand, int startTime, int endTime, int careTime) {
      this->demand = demand;
      this->startTime = startTime;
      this->endTime = endTime;
      this->careTime = careTime;
    }

    int getDemand() const { return demand; }
    int getStartTime() const { return startTime; }
    int getEndTime() const { return endTime; }
    int getCareTime() const { return careTime; }

    bool valid() {
      return demand >= 0 && startTime >= 0 && endTime >= 0 && careTime >= 0;
    }

    string toString() {
      ostringstream oss;
      oss << "Patient(" << "demand:" << demand << ", startTime:" << startTime << ", endTime:" << endTime << ", careTime:" << careTime << ")";
      return oss.str();
    }
};
