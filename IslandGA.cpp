#include "IslandGA.hpp"
#include <iostream>
#include <thread>

IslandGA::IslandGA(int migrationInterval, int migrationSize, int totalGenerations, vector<RuinAndRepair> islands): migrationInterval(migrationInterval), migrationSize(migrationSize), totalGenerations(totalGenerations), islands(islands) {}

void IslandGA::migrate() {
  for (size_t i = 0; i < islands.size(); i++) {
    size_t dest = (i + 1) % islands.size();

    auto emigrants = islands[i].getBestIndividuals(migrationSize);
    islands[dest].injectIndividuals(emigrants);
  }
}

void IslandGA::run() {
  for (RuinAndRepair& island : islands) island.initPopulation();

  for (int g = 0; g < totalGenerations; g += migrationInterval) {
    vector<thread> threads;
    for (auto& island : islands) {
      threads.emplace_back([&island, this]() {
          island.runGenerations(migrationInterval);
          });
    }
    cout << "Running threads" << endl;

    // Wait for all islands to finish before migrating
    for (auto& t : threads) t.join();

    cout << "Generation: " << g+migrationInterval << endl;
    for (auto& island : islands) {
      island.printPopulationStats();
    }
    migrate();
  }
  vector<Individual> solutions;
  for (auto& i : islands) {
    solutions.push_back(i.getBestSolution());
  }
  for (auto i : solutions) {
    cout << "Solution with fitness: " << i.getFitness() << endl;
  }
}

Individual IslandGA::getSolution() {
  vector<Individual> solutions;
  for (auto& i : islands) {
    solutions.push_back(i.getBestSolution());
  }
  sort(solutions.begin(), solutions.end(), [](const Individual& a, const Individual& b) {
      return a.getFitness() < b.getFitness();  
      });
  return solutions[0];
}
