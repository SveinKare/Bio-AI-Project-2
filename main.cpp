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
#include <queue>
#include <condition_variable>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

static constexpr int MAX_ISLANDS = 8;
static constexpr int MIN_ISLANDS = 2;
static constexpr int TOTAL_GENERATIONS = 5000;
static constexpr int MIGRATION_INTERVAL = 200;
static constexpr int REPORTING_INTERVAL = 200;
static constexpr int GLOBAL_STAGNATION_LIMIT = 500;
static constexpr double STAGNATION_IMPROVEMENT_THRESHOLD = 0.1;
static constexpr double RESTART_MIN_DESTRUCTION = 0.20;
static constexpr double RESTART_DESTRUCTION_RANGE = 0.40;
static constexpr int RESTART_ISLAND_FRACTION = 2;
static constexpr int RESTART_POPULATION_SEED_DIVISOR = 2;

void runIslandGA() {
  HomeCare hc;
  hc.init("../data/train_1.json");

  int hwThreads = static_cast<int>(thread::hardware_concurrency());
  int numIslands = min(hwThreads, MAX_ISLANDS);
  if (numIslands < MIN_ISLANDS) numIslands = MIN_ISLANDS;

#ifdef _OPENMP
  omp_set_num_threads(numIslands);
  cout << "Island Model GA: " << numIslands << " islands on "
    << hwThreads << " hardware threads" << endl;
#else
  cout << "Island Model GA: " << numIslands << " islands (sequential fallback)" << endl;
#endif

  struct IslandConfig { int pop; double mut; int elite; int tourn; int stag; double restart; double greedy; };
  vector<IslandConfig> configs = {
    {150, 0.05, 12, 15, 60, 0.3, 0.50},
    {120, 0.12,  8,  8, 30, 0.7, 0.25},
    {100, 0.30,  4,  3, 15, 0.9, 0.0},
    { 80, 0.50,  2,  2, 10, 0.95, 0.0},
    {150, 0.08, 15, 20, 50, 0.4, 0.40},
    {100, 0.20,  6,  5, 25, 0.6, 0.0},
    {120, 0.15, 10, 10, 35, 0.5, 0.10},
    { 80, 0.40,  2,  3, 10, 0.9, 0.0},
  };

  vector<GA> islands;
  islands.reserve(numIslands);
  for (int i = 0; i < numIslands; ++i) {
    const auto& cfg = configs[i % static_cast<int>(configs.size())];
    islands.emplace_back(&hc, cfg.pop, cfg.mut, cfg.elite, cfg.tourn, cfg.stag, cfg.restart, i, cfg.greedy);
  }

#pragma omp parallel for schedule(static)
  for (int i = 0; i < numIslands; ++i) {
    islands[i].initialize();
  }

  auto trueFitness = [&](const Individual& ind) -> double {
    FitnessResult result = islands[0].evaluateFitnessDetailed(ind);
    return result.rawTravelTime + result.penalty;
  };

  Individual globalBest = islands[0].getBest();
  double globalBestTrueFitness = trueFitness(globalBest);
  for (int i = 1; i < numIslands; ++i) {
    Individual islandBest = islands[i].getBest();
    if (islandBest.getGenes().empty()) continue;
    double fitness = trueFitness(islandBest);
    if (fitness < globalBestTrueFitness) {
      globalBest = islandBest;
      globalBestTrueFitness = fitness;
    }
  }

  double previousGlobalBest = globalBestTrueFitness;
  int globalStagnationCounter = 0;

  mt19937 migrationRng(42);

  auto startTime = chrono::steady_clock::now();

  for (int generation = 0; generation < TOTAL_GENERATIONS; ++generation) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < numIslands; ++i) {
      islands[i].runGeneration();
    }

    if ((generation + 1) % MIGRATION_INTERVAL == 0 && numIslands > 1) {
      vector<Individual> migrants(numIslands);
      for (int i = 0; i < numIslands; ++i) {
        migrants[i] = islands[i].getBest();
      }

      for (int i = 0; i < numIslands; ++i) {
        int destinationIsland = (i + 1) % numIslands;
        auto& destinationPopulation = islands[destinationIsland].getPopulation();
        int populationSize = static_cast<int>(destinationPopulation.size());
        int eliteCount = islands[destinationIsland].getEliteCount();
        if (populationSize <= eliteCount) continue;

        uniform_int_distribution<int> nonEliteDist(eliteCount, populationSize - 1);
        int replacementIdx = nonEliteDist(migrationRng);
        destinationPopulation[replacementIdx] = migrants[i];
      }
    }

    for (int i = 0; i < numIslands; ++i) {
      Individual islandBest = islands[i].getBest();
      if (islandBest.getGenes().empty()) continue;
      double fitness = trueFitness(islandBest);
      if (fitness < globalBestTrueFitness) {
        globalBest = islandBest;
        globalBestTrueFitness = fitness;
      }
    }

    if (globalBestTrueFitness < previousGlobalBest - STAGNATION_IMPROVEMENT_THRESHOLD) {
      previousGlobalBest = globalBestTrueFitness;
      globalStagnationCounter = 0;
    } else {
      globalStagnationCounter++;
    }

    if (globalStagnationCounter >= GLOBAL_STAGNATION_LIMIT) {
      vector<pair<double, int>> islandsByFitness;
      for (int i = 0; i < numIslands; ++i) {
        islandsByFitness.push_back({trueFitness(islands[i].getBest()), i});
      }
      sort(islandsByFitness.begin(), islandsByFitness.end());

      int islandsToRestart = numIslands / RESTART_ISLAND_FRACTION;
#pragma omp parallel for schedule(static)
      for (int j = numIslands - islandsToRestart; j < numIslands; ++j) {
        int islandIdx = islandsByFitness[j].second;
        islands[islandIdx].initialize();
        auto& islandPopulation = islands[islandIdx].getPopulation();
        int seedCount = static_cast<int>(islandPopulation.size()) / RESTART_POPULATION_SEED_DIVISOR;
        mt19937 restartRng(static_cast<unsigned>(
              7000000u + islandIdx * 999983u + generation * 104729u));
        for (int s = 0; s < seedCount && s < static_cast<int>(islandPopulation.size()); ++s) {
          islandPopulation[s] = globalBest;
          double destruction = RESTART_MIN_DESTRUCTION + RESTART_DESTRUCTION_RANGE * s / std::max(seedCount - 1, 1);
          islands[islandIdx].ruinAndRecreate(islandPopulation[s], destruction, restartRng);
        }
      }

      int bestIslandIdx = islandsByFitness[0].second;
      auto& bestIslandPopulation = islands[bestIslandIdx].getPopulation();
      if (!bestIslandPopulation.empty()) {
        bestIslandPopulation[0] = globalBest;
        bestIslandPopulation[0].setFitness(islands[bestIslandIdx].evaluateFitness(globalBest));
      }

      globalStagnationCounter = 0;
      auto elapsed = chrono::duration_cast<chrono::seconds>(
          chrono::steady_clock::now() - startTime).count();
      cout << "  >> R&R restart at gen " << (generation + 1)
        << " (" << elapsed << "s)" << endl;
    }

    if ((generation + 1) % REPORTING_INTERVAL == 0) {
      FitnessResult details = islands[0].evaluateFitnessDetailed(globalBest);
      auto elapsed = chrono::duration_cast<chrono::seconds>(
          chrono::steady_clock::now() - startTime).count();
      cout << "Gen " << setw(5) << (generation + 1)
        << " | true: " << fixed << setprecision(1) << globalBestTrueFitness;
      if (details.isValid) {
        cout << " | raw: " << details.rawTravelTime;
      } else {
        cout << " | raw: " << details.rawTravelTime << " (inv)";
      }
      cout << " | " << elapsed << "s" << endl;
    }
  }

  auto totalElapsed = chrono::duration_cast<chrono::milliseconds>(
      chrono::steady_clock::now() - startTime).count();

  FitnessResult finalDetails = islands[0].evaluateFitnessDetailed(globalBest);
  cout << "\n--- Final (Island Model, " << numIslands << " islands) ---" << endl;
  cout << "Wall time: " << fixed << setprecision(1) << totalElapsed / 1000.0 << "s" << endl;
  cout << "True fitness (raw + unscaled penalty): " << fixed << setprecision(1)
    << globalBestTrueFitness << endl;
  if (finalDetails.isValid) {
    cout << "Raw travel time (valid): " << finalDetails.rawTravelTime << endl;
  } else {
    cout << "Raw travel time: " << finalDetails.rawTravelTime
      << " (constraint violations, penalty=" << finalDetails.penalty << ")" << endl;
  }
  cout << "Benchmark: " << hc.getBenchmark() << endl;
}

void writeClustersToCSV(const std::vector<std::vector<int>>& clusters,
    const std::vector<Patient>& patients,
    const std::string& path) {
  std::ofstream file(path);
  if (!file.is_open()) {
    std::cerr << "Could not open file: " << path << std::endl;
    return;
  }

  file << "patient_index,x,y,cluster\n";
  for (int c = 0; c < (int)clusters.size(); c++) {
    for (int patientIdx : clusters[c]) {
      const Patient& p = patients[patientIdx];
      file << patientIdx << ","
        << p.getXCoord() << ","
        << p.getYCoord() << ","
        << c << "\n";
    }
  }

  file.close();
  std::cout << "Clusters written to " << path << std::endl;
}

void runParameterTuning() {
    vector<double> penaltyValues = {0.5, 1.0, 2.5, 5.0};
    vector<double> crossoverRateValues = {0.25, 0.5, 0.75, 1.0};
    vector<double> mutationRateValues = {0.25, 0.5, 0.75, 1.0};
    vector<double> scalingFactorValues = {0.5, 1.0, 2.5, 5.0};

    // Fixed parameters
    double epsilon = 0.1;
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
                popSize, epsilon, kParents, generations,
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

string getTimestamp() {
    auto now = chrono::system_clock::now();
    time_t t = chrono::system_clock::to_time_t(now);
    tm* tm_info = localtime(&t);
    char buf[20];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", tm_info);
    return string(buf);
}
// Output: 20260305_143201

void runRuinAndRepairGA() {
  HomeCare homeCare;
  homeCare.init("./data/test_instance_3.json");

  // ── Parameters ────────────────────────────────────────────────────────
  unsigned int seed        = random_device{}();
  int    numIslands        = 5;
  int    islandPopSize     = 1000;
  int    stage1Gens        = 3000;
  int    elitesPerIsland   = 200;
  double epsilon           = 0.1;
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

  ofstream file("./instance_3/" + getTimestamp() + ".txt");

  // ── Log parameters ────────────────────────────────────────────────────
  file << "=== Stage 1 Parameters ===" << "\n"
       << "Seed:            " << seed << "\n"
       << "numIslands:      " << numIslands      << "\n"
       << "islandPopSize:   " << islandPopSize   << "\n"
       << "stage1Gens:      " << stage1Gens      << "\n"
       << "elitesPerIsland: " << elitesPerIsland << "\n"
       << "epsilon:         " << epsilon         << "\n"
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

  // ── Stage 1 ───────────────────────────────────────────────────────────
  vector<Individual> elitePool;
  vector<RuinAndRepair> islands;

  for (int i = 0; i < numIslands; i++) {
    RuinAndRepair island(
        homeCare, islandPopSize, epsilon, kParents, stage1Gens,
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

  // ── Stage 2 ───────────────────────────────────────────────────────────
  RuinAndRepair exploitIsland(
      homeCare, (int)elitePool.size(), 0.0, exploitKParents, exploitGens,
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
  //runIslandGA();
  //runParameterTuning();
  for (int i = 0; i < 100; i++) {
    runRuinAndRepairGA();
  }
  return 0;
}


