#include "RuinAndRepair.hpp"
#include <vector>

using namespace std;

class IslandGA {
  private:
    vector<RuinAndRepair> islands;
    int migrationInterval;
    int migrationSize;
    int totalGenerations;
  public:
    IslandGA(int migrationInterval, int migrationSize, int totalGenerations, vector<RuinAndRepair> islands);
    void migrate();
    void run();
    Individual getSolution();
};
