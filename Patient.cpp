#include "Patient.hpp"
#include <sstream>

Patient::Patient() 
    : demand(-1), startTime(-1), endTime(-1), careTime(-1), xCoord(-1), yCoord(-1) {}

Patient::Patient(int demand, int startTime, int endTime, int careTime, int xCoord, int yCoord)
    : demand(demand), startTime(startTime), endTime(endTime), 
      careTime(careTime), xCoord(xCoord), yCoord(yCoord) {}

int Patient::getDemand() const { return demand; }
int Patient::getStartTime() const { return startTime; }
int Patient::getEndTime() const { return endTime; }
int Patient::getCareTime() const { return careTime; }
int Patient::getXCoord() const { return xCoord; }
int Patient::getYCoord() const { return yCoord; }

bool Patient::valid() {
    return demand >= 0 && startTime >= 0 && endTime >= 0 
        && careTime >= 0 && xCoord >= 0 && yCoord >= 0;
}

std::string Patient::toString() {
    std::ostringstream oss;
    oss << "Patient(demand:" << demand << ", startTime:" << startTime 
        << ", endTime:" << endTime << ", careTime:" << careTime 
        << ", x:" << xCoord << ", y:" << yCoord << ")";
    return oss.str();
}
