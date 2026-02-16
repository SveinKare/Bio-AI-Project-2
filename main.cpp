#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <thread>
#include <chrono>
#include <random>
#include "HomeCare.cpp"
#include "GA.h"

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

int main() {
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

    return 0;
}
