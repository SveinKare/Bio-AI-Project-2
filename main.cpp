#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>
#include <vector>
#include <algorithm>
#include <chrono>
#include "HomeCare.cpp"
#include "GA.h"

using namespace std;

static void writeSolution(const string& filename, const Individual& best,
                          const HomeCare& hc, const GA& ga) {
    const auto& genes = best.getGenes();
    const auto& patients = hc.getPatients();
    int capacity = hc.getCapacity();
    int returnTime = hc.getReturnTime();

    vector<vector<int>> routes;
    vector<int> currentRoute;
    for (int gene : genes) {
        if (gene == -1) {
            if (!currentRoute.empty()) {
                routes.push_back(std::move(currentRoute));
                currentRoute.clear();
            }
        } else if (gene > 0) {
            currentRoute.push_back(gene);
        }
    }
    if (!currentRoute.empty()) routes.push_back(std::move(currentRoute));

    ofstream out(filename);
    out << fixed << setprecision(1);

    out << "Nurse capacity: " << capacity << "\n";
    out << "Depot return time: " << returnTime << "\n";
    out << string(80, '-') << "\n";

    double totalDuration = 0;
    for (size_t i = 0; i < routes.size(); ++i) {
        const auto& route = routes[i];
        double duration = hc.getTravelTime(0, route[0]);
        double time = duration;
        int demand = 0;
        for (size_t j = 0; j < route.size(); ++j) {
            int p = route[j];
            demand += patients[p].getDemand();
            int earliest = patients[p].getStartTime();
            if (time < earliest) time = earliest;
            time += patients[p].getCareTime();
            if (j + 1 < route.size())  {
                double t = hc.getTravelTime(p, route[j + 1]);
                duration += t;
                time += t;
            }
        }
        double ret = hc.getTravelTime(route.back(), 0);
        duration += ret;
        totalDuration += duration;

        out << "Nurse " << (i + 1)
            << "\tDuration: " << duration
            << "\tDemand: " << demand
            << "\tPatients:";
        for (int p : route) out << " " << p;
        out << "\n";
    }

    out << string(80, '-') << "\n";
    out << "Objective value (total duration): " << totalDuration << "\n";
    out.close();

    cout << "Solution written to " << filename << endl;
}

static constexpr int TOTAL_GENERATIONS = 20000;
static constexpr int REPORTING_INTERVAL = 500;
static constexpr int POPULATION_SIZE = 400;
static constexpr int ELITE_COUNT = 5;
static constexpr int TOURNAMENT_SIZE = 5;
static constexpr double MUTATION_RATE = 0.01;
static constexpr double CROWDING_SF = 0.5;



int main() {
    HomeCare hc;
    hc.init("../data/train_2.json");

    GA ga(&hc, POPULATION_SIZE, MUTATION_RATE, ELITE_COUNT, TOURNAMENT_SIZE, CROWDING_SF);
    ga.initialize();

    auto trueFitness = [&](const Individual& ind) -> double {
        FitnessResult result = ga.evaluateFitnessDetailed(ind);
        return result.rawTravelTime + result.penalty;
    };

    Individual globalBest = ga.getBest();
    double globalBestTrueFitness = trueFitness(globalBest);

    bool hasFeasible = false;
    Individual bestFeasible(0, 0.0);
    double bestFeasibleFitness = numeric_limits<double>::max();

    auto startTime = chrono::steady_clock::now();

    for (int generation = 0; generation < TOTAL_GENERATIONS; ++generation) {
        ga.runGeneration();

        Individual currentBest = ga.getBest();
        double fitness = trueFitness(currentBest);
        if (fitness < globalBestTrueFitness) {
            globalBest = currentBest;
            globalBestTrueFitness = fitness;
        }

        for (const Individual& ind : ga.getPopulation()) {
            FitnessResult r = ga.evaluateFitnessDetailed(ind);
            if (r.isValid && r.rawTravelTime < bestFeasibleFitness) {
                bestFeasible = ind;
                bestFeasibleFitness = r.rawTravelTime;
                hasFeasible = true;
            }
        }

        if ((generation + 1) % REPORTING_INTERVAL == 0) {
            FitnessResult details = ga.evaluateFitnessDetailed(globalBest);
            auto elapsed = chrono::duration_cast<chrono::seconds>(
                chrono::steady_clock::now() - startTime).count();
            cout << "Gen " << setw(5) << (generation + 1)
                 << " | best: " << fixed << setprecision(1) << globalBestTrueFitness;
            if (details.isValid) {
                cout << " | raw: " << details.rawTravelTime;
            } else {
                cout << " | raw: " << details.rawTravelTime << " (inv)";
            }
            if (hasFeasible) {
                cout << " | feasible: " << bestFeasibleFitness;
            } else {
                cout << " | feasible: -";
            }
            cout << " | " << elapsed << "s" << endl;
        }
    }

    auto totalElapsed = chrono::duration_cast<chrono::milliseconds>(
        chrono::steady_clock::now() - startTime).count();

    FitnessResult finalDetails = ga.evaluateFitnessDetailed(globalBest);
    cout << "\n--- Final ---" << endl;
    cout << "Wall time: " << fixed << setprecision(1) << totalElapsed / 1000.0 << "s" << endl;
    cout << "Best overall (raw + penalty): " << fixed << setprecision(1)
         << globalBestTrueFitness << endl;
    if (finalDetails.isValid) {
        cout << "Best overall raw travel time (valid): " << finalDetails.rawTravelTime << endl;
    } else {
        cout << "Best overall raw travel time: " << finalDetails.rawTravelTime
             << " (constraint violations, penalty=" << finalDetails.penalty << ")" << endl;
    }
    if (hasFeasible) {
        cout << "Best feasible (output): " << bestFeasibleFitness << endl;
    } else {
        cout << "WARNING: No feasible solution found. Cannot write valid output." << endl;
    }
    cout << "Benchmark: " << hc.getBenchmark() << endl;

    if (hasFeasible) {
        writeSolution("../solution.txt", bestFeasible, hc, ga);
    }

    return 0;
}
