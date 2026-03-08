#pragma once
#include <string>

class Patient {
private:
    int demand;
    int startTime;
    int endTime;
    int careTime;
    int xCoord;
    int yCoord;

public:
    Patient();
    Patient(int demand, int startTime, int endTime, int careTime, int xCoord, int yCoord);

    int getDemand() const;
    int getStartTime() const;
    int getEndTime() const;
    int getCareTime() const;
    int getXCoord() const;
    int getYCoord() const;
    bool valid();
    std::string toString();
};
