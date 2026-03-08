#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>
#include <random>
#include "HomeCare.hpp"
#include "GA.h"
#include "RuinAndRepair.hpp"
#include "Similarity.hpp"
#include <fstream>
#include "IslandGA.hpp"
#include <mutex>
#include <atomic>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

void runParameterTuning() {
    vector<double> penaltyValues = {0.5, 1.0, 2.5, 5.0};
    vector<double> crossoverRateValues = {0.25, 0.5, 0.75, 1.0};
    vector<double> mutationRateValues = {0.25, 0.5, 0.75, 1.0};
    vector<double> scalingFactorValues = {0.5, 1.0, 2.5, 5.0};

    // Fixed parameters
    int popSize = 1000;
    int kParents = 3;
    int generations = 100;
    int kElites = 6;

    struct Result {
        double penalty, crossoverRate, mutationRate, scalingFactor;
        double minFitness;
        bool foundFeasible;
    };

    struct Task {
        double penalty, crossoverRate, mutationRate, scalingFactor;
    };

    // Build task queue
    vector<Task> tasks;
    for (double penalty : penaltyValues)
        for (double crossoverRate : crossoverRateValues)
            for (double mutationRate : mutationRateValues)
                for (double scalingFactor : scalingFactorValues)
                    tasks.push_back({penalty, crossoverRate, mutationRate, scalingFactor});

    const int total = tasks.size();
    vector<Result> results;
    results.reserve(total);

    mutex resultsMutex;
    mutex coutMutex;
    atomic<int> current{0};

    // Determine thread count (leave one core free)
    const int numThreads = 8;
    cout << "Running " << total << " configurations across " << numThreads << " threads\n";

    // Worker lambda
    auto worker = [&](int threadId, int start, int end) {
        for (int i = start; i < end; i++) {
            const Task& t = tasks[i];
            int runNum = ++current;

            {
                lock_guard<mutex> lock(coutMutex);
                cout << "Run " << runNum << "/" << total
                     << " | penalty=" << t.penalty
                     << " crossover=" << t.crossoverRate
                     << " mutation=" << t.mutationRate
                     << " scaling=" << t.scalingFactor
                     << " [thread " << threadId << "]\n";
            }

            HomeCare homeCare;
            homeCare.init("./data/train_0.json");
            RuinAndRepair r(
                homeCare,
                popSize, kParents, generations,
                t.penalty,   // Penalty
                t.crossoverRate,   // Crossover rate
                t.mutationRate,   // Mutation rate
                t.scalingFactor,   // Scaling factor
                kElites,     // k Elites
                cosineSimilarity,
                random_device{}()
                );
            try {
              r.run();
            } catch (const runtime_error& e) {
              cout << e.what() << endl;
              return;
            }


            auto solution = r.getBestSolution();
            double fitness = solution.getFitness();
            bool foundFeasible = (fitness != numeric_limits<double>::max());

            {
              lock_guard<mutex> lock(coutMutex);
              cout << "  [thread " << threadId << "] Run " << runNum
                << " fitness=" << fitness
                << " | Feasible: " << (foundFeasible ? "YES" : "NO") << "\n";
            }

            {
              lock_guard<mutex> lock(resultsMutex);
              results.push_back({t.penalty, t.crossoverRate, t.mutationRate,
                  t.scalingFactor, fitness, foundFeasible});
            }
        }
    };

    // Distribute tasks across threads
    vector<thread> threads;
    threads.reserve(numThreads);
    int chunkSize = (total + numThreads - 1) / numThreads;

    for (int i = 0; i < numThreads; i++) {
      int start = i * chunkSize;
      int end = min(start + chunkSize, total);
      if (start >= total) break;
      threads.emplace_back(worker, i, start, end);
    }

    for (auto& t : threads) t.join();

    // --- Post-processing (unchanged) ---
    vector<Result> feasibleResults, infeasibleResults;
    for (const auto& res : results) {
      if (res.foundFeasible) feasibleResults.push_back(res);
      else                   infeasibleResults.push_back(res);
    }

    sort(feasibleResults.begin(), feasibleResults.end(),
        [](const Result& a, const Result& b) {
        return a.minFitness < b.minFitness;
        });

    ofstream file("tuning.csv");
    file << "fitness,penalty,crossover_rate,mutation_rate,scaling_factor,feasible\n";
    file << fixed << setprecision(4);

    for (const auto& res : feasibleResults)
      file << res.minFitness << "," << res.penalty << "," << res.crossoverRate
        << "," << res.mutationRate << "," << res.scalingFactor << ",1\n";

    for (const auto& res : infeasibleResults)
      file << "N/A," << res.penalty << "," << res.crossoverRate
        << "," << res.mutationRate << "," << res.scalingFactor << ",0\n";

    file.close();
    cout << "\nResults written to file\n";

    cout << "\n=== Summary ===\n";
    cout << "Total runs: " << results.size() << "\n";
    cout << "Feasible solutions: " << feasibleResults.size() << "\n";
    cout << "Infeasible solutions: " << infeasibleResults.size() << "\n";

    if (!feasibleResults.empty()) {
      cout << "\n=== Top 5 Feasible Configurations ===\n";
      for (int i = 0; i < min(5, (int)feasibleResults.size()); i++) {
        const auto& res = feasibleResults[i];
        cout << "#" << i+1 << " fitness=" << res.minFitness
          << " | penalty=" << res.penalty
          << " | crossover=" << res.crossoverRate
          << " | mutation=" << res.mutationRate
          << " | scaling=" << res.scalingFactor << "\n";
      }
    } else {
      cout << "\nNo feasible solutions found!\n";
    }

    if (!infeasibleResults.empty())
      cout << "\n" << infeasibleResults.size()
        << " configurations did not find any feasible solution.\n";
}

void measureEntropy() {
  HomeCare homeCare;
  homeCare.init("./data/test_instance_3.json");
  unsigned int seed        = random_device{}();
  ofstream file("entropy_logging_exploit_island.txt");
  int    numIslands        = 5;
  int    islandPopSize     = 1000;
  int    gens              = 3000;
  int    kParents          = 20;
  double penalty           = 2.5;
  double crossoverRate     = 0.5;
  double mutationRate      = 0.5;
  double scalingFactor     = 2.5;
  int    kElites           = 20;

  RuinAndRepair island(
      homeCare, islandPopSize, kParents, gens,
      penalty, crossoverRate, mutationRate, scalingFactor, kElites,
      cosineSimilarity,seed
      );
  island.initPopulation();
  for (int i = 0; i < gens; i++) {
    island.runGenerations(1);
    file << island.edgeEntropy() << " ";
  }
  file << "\n";
  file.close();
}

string getTimestamp() {
  auto now = chrono::system_clock::now();
  time_t t = chrono::system_clock::to_time_t(now);
  tm* tm_info = localtime(&t);
  char buf[20];
  strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", tm_info);
  return string(buf);
}

void runRuinAndRepairGA() {
  HomeCare homeCare;
  homeCare.init("./data/test_instance_3.json");

  // Parameters
  unsigned int seed        = random_device{}();
  int    numIslands        = 5;
  int    islandPopSize     = 1000;
  int    stage1Gens        = 3000;
  int    elitesPerIsland   = 200;
  int    kParents          = 3;
  double penalty           = 0.05;
  double crossoverRate     = 0.9;
  double mutationRate      = 0.3;
  double scalingFactor     = 2.5;
  int    kElites           = 6;

  int    exploitKParents   = 20;
  double exploitPenalty    = 2.5;
  int    exploitGens       = 3000;
  double exploitCrossover  = 0.5;
  double exploitMutation   = 0.5;
  int    exploitKElites    = 20;

  ofstream file(getTimestamp() + ".txt");

  // Logging parameters 
  file << "=== Stage 1 Parameters ===" << "\n"
    << "Seed:            " << seed << "\n"
    << "numIslands:      " << numIslands      << "\n"
    << "islandPopSize:   " << islandPopSize   << "\n"
    << "stage1Gens:      " << stage1Gens      << "\n"
    << "elitesPerIsland: " << elitesPerIsland << "\n"
    << "kParents:        " << kParents        << "\n"
    << "penalty:         " << penalty         << "\n"
    << "crossoverRate:   " << crossoverRate   << "\n"
    << "mutationRate:    " << mutationRate    << "\n"
    << "scalingFactor:   " << scalingFactor   << "\n"
    << "kElites:         " << kElites         << "\n"
    << "\n=== Stage 2 Parameters ===" << "\n"
    << "exploitPopSize:  " << numIslands * elitesPerIsland << "\n"
    << "exploitKParents: " << exploitKParents << "\n"
    << "exploitPenalty:  " << exploitPenalty << "\n"
    << "exploitGens:     " << exploitGens     << "\n"
    << "exploitCrossover:" << exploitCrossover << "\n"
    << "exploitMutation: " << exploitMutation << "\n"
    << "exploitKElites:  " << exploitKElites  << "\n"
    << "\n=== Results ===" << "\n";

  // Stage 1
  vector<Individual> elitePool;
  vector<RuinAndRepair> islands;

  for (int i = 0; i < numIslands; i++) {
    RuinAndRepair island(
        homeCare, islandPopSize, kParents, stage1Gens,
        penalty, crossoverRate, mutationRate, scalingFactor, kElites,
        cosineSimilarity,(unsigned int)(seed+i)
        );
    islands.emplace_back(island);
    islands.back().initPopulation();
  }

  vector<thread> threads;
  for (auto& island : islands) {
    threads.emplace_back([&island, stage1Gens]() {
        island.runGenerations(stage1Gens);
        });
  }
  for (auto& t : threads) t.join();

  for (auto& island : islands) {
    auto elites = island.getBestIndividuals(elitesPerIsland);
    elitePool.insert(elitePool.end(), elites.begin(), elites.end());
  }

  // Stage 2
  RuinAndRepair exploitIsland(
      homeCare, (int)elitePool.size(), exploitKParents, exploitGens,
      exploitPenalty, exploitCrossover, exploitMutation, scalingFactor, exploitKElites,
      cosineSimilarity, (unsigned int)(seed+numIslands)
      );
  exploitIsland.setPopulation(elitePool);
  exploitIsland.runGenerations(exploitGens);

  auto solution = exploitIsland.getBestSolution();
  auto doubleCheck = homeCare.calculateFitness(solution.getGenes(), 1.0);
  if (doubleCheck.second == 0.0) {
    cout << "Valid solution!" << endl;
  } else {
    cout << "Penalty was: " << doubleCheck.second << endl;
  }
  cout << "Benchmark: " << homeCare.getBenchmark() << endl;
  cout  << "Solution: " << solution.getFitness() << endl;
  file  << "Solution: " << solution.getFitness() << "\n";
  for (auto& g : solution.getGenes()) {
    file << g << " ";
  }
  file << "\n";
  auto output = homeCare.printSolution(solution.getGenes());
  cout << output << endl;
  file << output << "\n";
  file.close();
}


int main() {
  runRuinAndRepairGA();
  return 0;
}


