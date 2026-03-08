#include "GA.h"
#include "HomeCare.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

static constexpr int DEPOT = -1;
static constexpr int ROUTE_SEPARATOR = -1;

static constexpr double LATE_PENALTY_WEIGHT = 10000.0;
static constexpr double CAPACITY_PENALTY_WEIGHT = 1000.0;
static constexpr double INITIAL_PENALTY_MULTIPLIER = 0.1;
static constexpr double MIN_PENALTY_MULTIPLIER = 0.01;
static constexpr double MAX_PENALTY_MULTIPLIER = 10.0;
static constexpr double PENALTY_DECREASE_RATE = 0.95;
static constexpr double PENALTY_INCREASE_RATE = 1.05;
static constexpr double FEASIBILITY_UPPER_BOUND = 0.6;
static constexpr double FEASIBILITY_LOWER_BOUND = 0.3;

static constexpr int TWO_OPT_MAX_PASSES = 3;
static constexpr int OR_OPT_MAX_SEGMENT_LENGTH = 3;
static constexpr int LOCAL_SEARCH_MAX_INDIVIDUALS = 2;
static constexpr int LOCAL_SEARCH_GENERATION_INTERVAL = 3;

static constexpr double MAX_BOOSTED_MUTATION_RATE = 0.35;
static constexpr double MUTATION_BOOST_FACTOR = 2.0;

static constexpr double BASE_RUIN_RECREATE_PROBABILITY = 0.15;
static constexpr double STAGNATION_RUIN_RECREATE_PROBABILITY = 0.40;
static constexpr double BASE_DESTRUCTION_RATE = 0.25;
static constexpr double STAGNATION_DESTRUCTION_RATE = 0.45;

static constexpr double WORST_REMOVAL_PROBABILITY = 0.5;
static constexpr double WORST_REMOVAL_RANDOMIZATION = 3.0;
static constexpr double WORST_REMOVAL_ACCEPTANCE = 0.9;

static constexpr double GREEDY_RESTART_FRACTION = 0.25;

static mt19937 ga_rng(1234);

static vector<vector<int>> genomeToRoutes(const vector<int>& genes) {
    vector<vector<int>> routes;
    vector<int> currentRoute;
    for (int gene : genes) {
        if (gene == ROUTE_SEPARATOR) {
            if (!currentRoute.empty()) {
                routes.push_back(std::move(currentRoute));
                currentRoute.clear();
            }
        } else if (gene >= 0) {
            currentRoute.push_back(gene);
        }
    }
    if (!currentRoute.empty()) routes.push_back(std::move(currentRoute));
    return routes;
}

static vector<int> routesToGenome(const vector<vector<int>>& routes) {
    vector<int> genome;
    for (size_t i = 0; i < routes.size(); ++i) {
        if (i > 0) genome.push_back(ROUTE_SEPARATOR);
        for (int patient : routes[i]) genome.push_back(patient);
    }
    return genome;
}

static double routeTravelCost(const vector<int>& route, const HomeCare* problem) {
    if (route.empty()) return 0;
    double cost = problem->getTravelTime(DEPOT, route[0]);
    for (size_t i = 1; i < route.size(); ++i)
        cost += problem->getTravelTime(route[i - 1], route[i]);
    cost += problem->getTravelTime(route.back(), DEPOT);
    return cost;
}

static double insertionCostDelta(const vector<int>& route, size_t pos, int patient, const HomeCare* problem) {
    int prev = (pos > 0) ? route[pos - 1] : DEPOT;
    int next = (pos < route.size()) ? route[pos] : DEPOT;
    return problem->getTravelTime(prev, patient)
         + problem->getTravelTime(patient, next)
         - problem->getTravelTime(prev, next);
}

static void insertPatientCheapest(vector<vector<int>>& routes, int patient, const HomeCare* problem) {
    if (routes.empty()) {
        routes.push_back({patient});
        return;
    }
    double bestDelta = numeric_limits<double>::max();
    size_t bestRoute = 0, bestPosition = 0;
    for (size_t r = 0; r < routes.size(); ++r) {
        for (size_t pos = 0; pos <= routes[r].size(); ++pos) {
            double delta = insertionCostDelta(routes[r], pos, patient, problem);
            if (delta < bestDelta) {
                bestDelta = delta;
                bestRoute = r;
                bestPosition = pos;
            }
        }
    }
    routes[bestRoute].insert(routes[bestRoute].begin() + static_cast<ptrdiff_t>(bestPosition), patient);
}

GA::GA(HomeCare* hc, int popSize, double mutRate, int elites, int tournSize,
       int stagnationGen, double restartFrac, int island, double greedyFrac)
    : problem(hc), populationSize(popSize), mutationRate(mutRate),
      eliteCount(elites), tournamentSize(tournSize),
      lastBestFitness(numeric_limits<double>::max()), generationsWithoutImprovement(0),
      stagnationThreshold(stagnationGen), restartFraction(restartFrac),
      generationCount(0), islandId(island), penaltyMultiplier(INITIAL_PENALTY_MULTIPLIER),
      greedyFraction(greedyFrac) {}

int GA::tournamentSelect() const {
    uniform_int_distribution<size_t> dist(0, population.size() - 1);
    int best = -1;
    double bestFitness = numeric_limits<double>::max();
    for (int k = 0; k < tournamentSize; ++k) {
        size_t idx = dist(ga_rng);
        double f = population[idx].getFitness();
        if (f < bestFitness) {
            bestFitness = f;
            best = static_cast<int>(idx);
        }
    }
    return best;
}

int GA::tournamentSelect(mt19937& rng) const {
    uniform_int_distribution<size_t> dist(0, population.size() - 1);
    int best = -1;
    double bestFitness = numeric_limits<double>::max();
    for (int k = 0; k < tournamentSize; ++k) {
        size_t idx = dist(rng);
        double f = population[idx].getFitness();
        if (f < bestFitness) {
            bestFitness = f;
            best = static_cast<int>(idx);
        }
    }
    return best;
}

void GA::apply2Opt(Individual& ind) const {
    vector<vector<int>> routes = genomeToRoutes(ind.getGenes());
    for (auto& route : routes) {
        int routeLength = static_cast<int>(route.size());
        if (routeLength < 2) continue;
        for (int pass = 0; pass < TWO_OPT_MAX_PASSES; ++pass) {
            bool improved = false;
            for (int i = 0; i < routeLength - 1 && !improved; ++i) {
                for (int j = i + 1; j < routeLength && !improved; ++j) {
                    int beforeSegment = (i > 0) ? route[i - 1] : DEPOT;
                    int afterSegment = (j < routeLength - 1) ? route[j + 1] : DEPOT;
                    double currentCost = problem->getTravelTime(beforeSegment, route[i])
                                       + problem->getTravelTime(route[j], afterSegment);
                    double reversedCost = problem->getTravelTime(beforeSegment, route[j])
                                        + problem->getTravelTime(route[i], afterSegment);
                    if (reversedCost < currentCost) {
                        std::reverse(route.begin() + i, route.begin() + j + 1);
                        improved = true;
                    }
                }
            }
            if (!improved) break;
        }
    }
    ind.setGenes(routesToGenome(routes));
}

void GA::applyOrOpt(Individual& ind) const {
    vector<vector<int>> routes = genomeToRoutes(ind.getGenes());
    if (routes.size() < 2) return;
    double bestFitness = evaluateRoutes(routes);

    for (size_t srcRoute = 0; srcRoute < routes.size(); ++srcRoute) {
        for (int segLen = 1; segLen <= OR_OPT_MAX_SEGMENT_LENGTH && segLen <= static_cast<int>(routes[srcRoute].size()); ++segLen) {
            for (size_t segStart = 0; segStart + segLen <= routes[srcRoute].size(); ++segStart) {
                vector<int> segment(routes[srcRoute].begin() + static_cast<ptrdiff_t>(segStart),
                                   routes[srcRoute].begin() + static_cast<ptrdiff_t>(segStart + segLen));

                routes[srcRoute].erase(routes[srcRoute].begin() + static_cast<ptrdiff_t>(segStart),
                                      routes[srcRoute].begin() + static_cast<ptrdiff_t>(segStart + segLen));

                for (size_t dstRoute = 0; dstRoute < routes.size(); ++dstRoute) {
                    if (dstRoute == srcRoute) continue;
                    for (size_t insertPos = 0; insertPos <= routes[dstRoute].size(); ++insertPos) {
                        routes[dstRoute].insert(routes[dstRoute].begin() + static_cast<ptrdiff_t>(insertPos),
                                               segment.begin(), segment.end());
                        double fitness = evaluateRoutes(routes);
                        if (fitness < bestFitness) {
                            if (routes[srcRoute].empty())
                                routes.erase(routes.begin() + static_cast<ptrdiff_t>(srcRoute));
                            ind.setGenes(routesToGenome(routes));
                            ind.setFitness(fitness);
                            return;
                        }
                        routes[dstRoute].erase(routes[dstRoute].begin() + static_cast<ptrdiff_t>(insertPos),
                                              routes[dstRoute].begin() + static_cast<ptrdiff_t>(insertPos + segLen));
                    }
                }

                routes[srcRoute].insert(routes[srcRoute].begin() + static_cast<ptrdiff_t>(segStart),
                                       segment.begin(), segment.end());
            }
        }
    }
}

void GA::applyRelocate(Individual& ind) const {
    vector<vector<int>> routes = genomeToRoutes(ind.getGenes());
    if (routes.size() < 2) return;
    double bestFitness = evaluateRoutes(routes);

    for (size_t srcRoute = 0; srcRoute < routes.size(); ++srcRoute) {
        for (size_t patientPos = 0; patientPos < routes[srcRoute].size(); ++patientPos) {
            int patient = routes[srcRoute][patientPos];

            routes[srcRoute].erase(routes[srcRoute].begin() + static_cast<ptrdiff_t>(patientPos));

            for (size_t dstRoute = 0; dstRoute < routes.size(); ++dstRoute) {
                if (dstRoute == srcRoute) continue;
                for (size_t insertPos = 0; insertPos <= routes[dstRoute].size(); ++insertPos) {
                    routes[dstRoute].insert(routes[dstRoute].begin() + static_cast<ptrdiff_t>(insertPos), patient);
                    double fitness = evaluateRoutes(routes);
                    if (fitness < bestFitness) {
                        if (routes[srcRoute].empty())
                            routes.erase(routes.begin() + static_cast<ptrdiff_t>(srcRoute));
                        ind.setGenes(routesToGenome(routes));
                        ind.setFitness(fitness);
                        return;
                    }
                    routes[dstRoute].erase(routes[dstRoute].begin() + static_cast<ptrdiff_t>(insertPos));
                }
            }

            routes[srcRoute].insert(routes[srcRoute].begin() + static_cast<ptrdiff_t>(patientPos), patient);
        }
    }
}

void GA::applyLocalSearch(Individual& ind) const {
    apply2Opt(ind);
    applyOrOpt(ind);
    applyRelocate(ind);
}

void GA::ruinAndRecreate(Individual& ind, double destructionRate, mt19937& rng) const {
    vector<vector<int>> routes = genomeToRoutes(ind.getGenes());
    int totalPatients = problem->getNbrPatients();
    int patientsToRemove = std::max(1, static_cast<int>(totalPatients * destructionRate));

    vector<pair<int, int>> patientLocations;
    for (int r = 0; r < static_cast<int>(routes.size()); ++r)
        for (int p = 0; p < static_cast<int>(routes[r].size()); ++p)
            patientLocations.push_back({r, p});

    uniform_real_distribution<double> unitDist(0.0, 1.0);
    vector<int> removedPatients;
    vector<bool> isRemoved(totalPatients, false);

    if (unitDist(rng) < WORST_REMOVAL_PROBABILITY) {
        vector<pair<double, pair<int, int>>> removalCosts;
        for (auto& [routeIdx, posIdx] : patientLocations) {
            int patient = routes[routeIdx][posIdx];
            int prev = (posIdx > 0) ? routes[routeIdx][posIdx - 1] : DEPOT;
            int next = (posIdx + 1 < static_cast<int>(routes[routeIdx].size())) ? routes[routeIdx][posIdx + 1] : DEPOT;
            double detourCost = problem->getTravelTime(prev, patient)
                              + problem->getTravelTime(patient, next)
                              - problem->getTravelTime(prev, next);
            removalCosts.push_back({-detourCost, {routeIdx, posIdx}});
        }
        sort(removalCosts.begin(), removalCosts.end());

        int picked = 0;
        for (auto& [negDetour, location] : removalCosts) {
            if (picked >= patientsToRemove) break;
            int patient = routes[location.first][location.second];
            if (isRemoved[patient]) continue;
            double randomizedAcceptance = std::pow(unitDist(rng), WORST_REMOVAL_RANDOMIZATION);
            if (randomizedAcceptance < WORST_REMOVAL_ACCEPTANCE) {
                isRemoved[patient] = true;
                removedPatients.push_back(patient);
                picked++;
            }
        }
        while (picked < patientsToRemove) {
            int randomPatient = uniform_int_distribution<int>(0, totalPatients - 1)(rng);
            if (!isRemoved[randomPatient]) {
                isRemoved[randomPatient] = true;
                removedPatients.push_back(randomPatient);
                picked++;
            }
        }
    } else {
        int seedRouteIdx = patientLocations[uniform_int_distribution<int>(0, static_cast<int>(patientLocations.size()) - 1)(rng)].first;
        int seedPatient = routes[seedRouteIdx].empty() ? 0 : routes[seedRouteIdx][0];

        vector<pair<double, int>> distancesToSeed;
        for (auto& [routeIdx, posIdx] : patientLocations) {
            int patient = routes[routeIdx][posIdx];
            double roundTripDistance = problem->getTravelTime(seedPatient, patient)
                                    + problem->getTravelTime(patient, seedPatient);
            distancesToSeed.push_back({roundTripDistance, patient});
        }
        sort(distancesToSeed.begin(), distancesToSeed.end());

        int picked = 0;
        for (auto& [distance, patient] : distancesToSeed) {
            if (picked >= patientsToRemove) break;
            if (!isRemoved[patient]) {
                isRemoved[patient] = true;
                removedPatients.push_back(patient);
                picked++;
            }
        }
    }

    for (auto& route : routes) {
        route.erase(
            std::remove_if(route.begin(), route.end(),
                [&](int patient) { return isRemoved[patient]; }),
            route.end());
    }
    routes.erase(
        std::remove_if(routes.begin(), routes.end(),
            [](const vector<int>& route) { return route.empty(); }),
        routes.end());
    if (routes.empty()) routes.push_back({});

    shuffle(removedPatients.begin(), removedPatients.end(), rng);
    for (int patient : removedPatients) {
        insertPatientCheapest(routes, patient, problem);
    }

    ind.setGenes(routesToGenome(routes));
    ind.setFitness(evaluateFitness(ind));
}

int GA::similarity(const Individual& a, const Individual& b) const {
    int totalPatients = problem->getNbrPatients();
    vector<int> routeAssignmentA(totalPatients, -1);
    vector<int> routeAssignmentB(totalPatients, -1);

    int routeIndex = 0;
    for (int gene : a.getGenes()) {
        if (gene == ROUTE_SEPARATOR) ++routeIndex;
        else if (gene >= 0 && gene < totalPatients) routeAssignmentA[gene] = routeIndex;
    }
    routeIndex = 0;
    for (int gene : b.getGenes()) {
        if (gene == ROUTE_SEPARATOR) ++routeIndex;
        else if (gene >= 0 && gene < totalPatients) routeAssignmentB[gene] = routeIndex;
    }

    int sharedAssignments = 0;
    for (int patient = 0; patient < totalPatients; ++patient)
        if (routeAssignmentA[patient] >= 0 && routeAssignmentA[patient] == routeAssignmentB[patient])
            ++sharedAssignments;
    return sharedAssignments;
}

void GA::runGeneration() {
    if (population.empty()) return;
    generationCount++;

    bool isStagnating = generationsWithoutImprovement > stagnationThreshold / 2;

    double effectiveMutationRate = mutationRate;
    if (isStagnating) {
        effectiveMutationRate = std::min(MAX_BOOSTED_MUTATION_RATE, mutationRate * MUTATION_BOOST_FACTOR);
    }

    evaluatePopulation();
    sort(population.begin(), population.end(),
         [](const Individual& a, const Individual& b) {
             return a.getFitness() < b.getFitness();
         });

    int actualElites = std::min(eliteCount, populationSize);

    vector<Individual> elites(actualElites);
    for (int i = 0; i < actualElites; ++i) {
        elites[i] = population[i];
    }

    bool localSearchThisGeneration = (generationCount % LOCAL_SEARCH_GENERATION_INTERVAL == 0);
    int localSearchCount = localSearchThisGeneration ? std::min(LOCAL_SEARCH_MAX_INDIVIDUALS, actualElites) : 0;
    for (int i = 0; i < actualElites; ++i) {
        if (i < localSearchCount) applyLocalSearch(elites[i]);
        elites[i].setFitness(evaluateFitness(elites[i]));
    }

    int childrenCount = populationSize - actualElites;
    vector<Individual> children(childrenCount);

    {
        mt19937 offspringRng(static_cast<unsigned>(
            2000000u + islandId * 999983u + generationCount * 104729u));

        double ruinRecreateProbability = isStagnating ? STAGNATION_RUIN_RECREATE_PROBABILITY : BASE_RUIN_RECREATE_PROBABILITY;
        double ruinRecreateDestruction = isStagnating ? STAGNATION_DESTRUCTION_RATE : BASE_DESTRUCTION_RATE;

        uniform_real_distribution<double> unitDist(0.0, 1.0);

        for (int c = 0; c < childrenCount; ++c) {
            int parentIdx1 = tournamentSelect(offspringRng);
            int parentIdx2 = parentIdx1;
            Individual child(0, 0.0);

            if (unitDist(offspringRng) < ruinRecreateProbability) {
                child = population[parentIdx1];
                ruinAndRecreate(child, ruinRecreateDestruction, offspringRng);
            } else {
                parentIdx2 = tournamentSelect(offspringRng);
                while (parentIdx2 == parentIdx1) parentIdx2 = tournamentSelect(offspringRng);
                child = crossoverFlat(population[parentIdx1], population[parentIdx2]);
                child.mutation(effectiveMutationRate, offspringRng);
            }
            child.setFitness(evaluateFitness(child));

            const Individual& parent1 = population[parentIdx1];
            const Individual& parent2 = population[parentIdx2];
            int similarityToParent1 = similarity(child, parent1);
            int similarityToParent2 = (parentIdx1 != parentIdx2) ? similarity(child, parent2) : similarityToParent1;
            const Individual& closerParent = (similarityToParent1 >= similarityToParent2) ? parent1 : parent2;

            double childFitness = child.getFitness();
            double parentFitness = closerParent.getFitness();
            double fitnessSum = childFitness + parentFitness;
            if (fitnessSum <= 0) fitnessSum = numeric_limits<double>::epsilon();
            double childSelectionProbability = parentFitness / fitnessSum;

            if (unitDist(offspringRng) < childSelectionProbability) {
                children[c] = std::move(child);
            } else {
                children[c] = closerParent;
            }
        }
    }

    vector<Individual> nextGeneration;
    nextGeneration.reserve(populationSize);
    for (auto& elite : elites) nextGeneration.push_back(std::move(elite));
    for (auto& child : children) nextGeneration.push_back(std::move(child));

    population = std::move(nextGeneration);

    double currentBestFitness = getBest().getFitness();
    if (currentBestFitness < lastBestFitness) {
        lastBestFitness = currentBestFitness;
        generationsWithoutImprovement = 0;
    } else {
        generationsWithoutImprovement++;
        if (generationsWithoutImprovement >= stagnationThreshold) {
            restartPartOfPopulation();
            generationsWithoutImprovement = 0;
        }
    }
}

Individual GA::createGreedyIndividual() const {
    return createGreedyIndividual(ga_rng);
}

Individual GA::createGreedyIndividual(std::mt19937& rng) const {
    int n = problem->getNbrPatients();
    int nbrNurses = problem->getNumberOfNurses();
    const auto& patients = problem->getPatients();
    int capacity = problem->getCapacity();
    int returnTime = problem->getReturnTime();
    vector<bool> assigned(static_cast<size_t>(n), false);
    vector<vector<int>> routes;

    int prev = DEPOT;
    double currentTime = 0;
    int currentStrain = 0;
    vector<int> currentRoute;

    vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;

    while (true) {
        shuffle(order.begin(), order.end(), rng);
        double bestCost = numeric_limits<double>::max();
        int bestP = -1;
        for (int idx = 0; idx < n; ++idx) {
            int p = order[idx];
            if (assigned[p]) continue;
            int demand = patients[p].getDemand();
            int careTime = patients[p].getCareTime();
            int earliest = patients[p].getStartTime();
            int latest = patients[p].getEndTime();
            if (currentStrain + demand > capacity) continue;

            double travel = problem->getTravelTime(prev, p);
            double arrival = currentTime + travel;
            if (arrival > latest) continue;
            double startTime = (arrival < earliest) ? earliest : arrival;
            double finishTime = startTime + careTime;
            if (finishTime + problem->getTravelTime(p, DEPOT) > returnTime) continue;

            if (travel < bestCost) { bestCost = travel; bestP = p; }
        }

        if (bestP < 0) {
            if (!currentRoute.empty()) {
                routes.push_back(std::move(currentRoute));
                currentRoute.clear();
                prev = DEPOT;
                currentTime = 0;
                currentStrain = 0;
                if (static_cast<int>(routes.size()) >= nbrNurses) break;
            } else {
                break;
            }
        } else {
            currentRoute.push_back(bestP);
            assigned[bestP] = true;
            double travel = problem->getTravelTime(prev, bestP);
            double arrival = currentTime + travel;
            int earliest = patients[bestP].getStartTime();
            int careTime = patients[bestP].getCareTime();
            currentTime = (arrival < earliest ? earliest : arrival) + careTime;
            currentStrain += patients[bestP].getDemand();
            prev = bestP;
        }
    }

    for (int p = 0; p < n; ++p) {
        if (!assigned[p]) insertPatientCheapest(routes, p, problem);
    }

    vector<int> genome = routesToGenome(routes);
    Individual ind(static_cast<int>(genome.size()), 0.0);
    ind.setGenes(genome);
    ind.setFitness(evaluateFitness(ind));
    return ind;
}

Individual GA::createRandomIndividual() const {
    return createRandomIndividual(ga_rng);
}

Individual GA::createRandomIndividual(mt19937& rng) const {
    int n = problem->getNbrPatients();
    int nbrNurses = problem->getNumberOfNurses();
    vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;
    shuffle(indices.begin(), indices.end(), rng);

    int numRoutes = uniform_int_distribution<int>(1, nbrNurses)(rng);
    vector<vector<int>> routes(numRoutes);
    for (int j = 0; j < n; ++j) {
        routes[j % numRoutes].push_back(indices[j]);
    }
    vector<int> genome = routesToGenome(routes);
    Individual ind(static_cast<int>(genome.size()), 0.0);
    ind.setGenes(genome);
    ind.setFitness(evaluateFitness(ind));
    return ind;
}

void GA::restartPartOfPopulation() {
    evaluatePopulation();
    sort(population.begin(), population.end(),
         [](const Individual& a, const Individual& b) {
             return a.getFitness() < b.getFitness();
         });
    int keepCount = static_cast<int>(populationSize * (1.0 - restartFraction));
    keepCount = std::max(keepCount, eliteCount);
    int restartedCount = populationSize - keepCount;
    int nGreedy = std::max(1, static_cast<int>(restartedCount * GREEDY_RESTART_FRACTION));
    int greedyEnd = std::min(keepCount + nGreedy, populationSize);

    mt19937 local_rng(static_cast<unsigned>(
        4000000u + generationCount * 104729u + islandId * 999983u));

    for (int i = keepCount; i < populationSize; ++i) {
        if (i < greedyEnd) {
            population[i] = createGreedyIndividual(local_rng);
        } else {
            population[i] = createRandomIndividual(local_rng);
        }
    }
}

void GA::initialize() {
    population.clear();
    lastBestFitness = numeric_limits<double>::max();
    generationsWithoutImprovement = 0;
    generationCount = 0;
    penaltyMultiplier = INITIAL_PENALTY_MULTIPLIER;
    int n = problem->getNbrPatients();
    if (n == 0) return;

    population.resize(populationSize);
    const int greedyCount = static_cast<int>(populationSize * greedyFraction);

    mt19937 local_rng(static_cast<unsigned>(3000000u + islandId * 999983u));

    for (int i = 0; i < greedyCount; ++i) {
        population[i] = createGreedyIndividual(local_rng);
    }
    for (int i = greedyCount; i < populationSize; ++i) {
        population[i] = createRandomIndividual(local_rng);
    }
}

Individual GA::crossoverFlat(const Individual& parent1, const Individual& parent2) const {
    auto routes1 = genomeToRoutes(parent1.getGenes());
    auto routes2 = genomeToRoutes(parent2.getGenes());
    int nbrPatients = problem->getNbrPatients();
    int maxNurses = problem->getNumberOfNurses();

    struct ScoredRoute { vector<int> route; double costPerPatient; };
    vector<ScoredRoute> all;
    for (const auto& r : routes1) {
        if (!r.empty())
            all.push_back({r, routeTravelCost(r, problem) / static_cast<double>(r.size())});
    }
    for (const auto& r : routes2) {
        if (!r.empty())
            all.push_back({r, routeTravelCost(r, problem) / static_cast<double>(r.size())});
    }
    sort(all.begin(), all.end(), [](const ScoredRoute& a, const ScoredRoute& b) {
        return a.costPerPatient < b.costPerPatient;
    });

    vector<vector<int>> childRoutes;
    vector<bool> assigned(static_cast<size_t>(nbrPatients), false);

    for (const auto& sr : all) {
        bool ok = true;
        for (int p : sr.route) if (assigned[p]) { ok = false; break; }
        if (ok) {
            childRoutes.push_back(sr.route);
            for (int p : sr.route) assigned[p] = true;
        }
    }

    for (int p = 0; p < nbrPatients; ++p) {
        if (!assigned[p]) insertPatientCheapest(childRoutes, p, problem);
    }

    while (static_cast<int>(childRoutes.size()) > maxNurses) {
        size_t a = 0, b = 1;
        if (childRoutes[1].size() < childRoutes[0].size()) std::swap(a, b);
        for (size_t i = 2; i < childRoutes.size(); ++i) {
            if (childRoutes[i].size() < childRoutes[a].size()) {
                b = a; a = i;
            } else if (childRoutes[i].size() < childRoutes[b].size()) {
                b = i;
            }
        }
        childRoutes[a].insert(childRoutes[a].end(), childRoutes[b].begin(), childRoutes[b].end());
        childRoutes.erase(childRoutes.begin() + static_cast<ptrdiff_t>(b));
    }

    vector<int> genome = routesToGenome(childRoutes);
    Individual child(static_cast<int>(genome.size()), 0.0);
    child.setGenes(genome);
    return child;
}

double GA::evaluateRoutes(const vector<vector<int>>& routes) const {
    const auto& patients = problem->getPatients();
    int capacity = problem->getCapacity();
    int returnTime = problem->getReturnTime();
    int maxNurses = problem->getNumberOfNurses();

    double totalTravel = 0;
    double penalty = 0;
    int activeRouteCount = 0;

    for (const auto& route : routes) {
        if (route.empty()) continue;
        ++activeRouteCount;
        int prev = DEPOT;
        double currentTime = 0;
        int currentStrain = 0;

        for (int p : route) {
            double travel = problem->getTravelTime(prev, p);
            totalTravel += travel;
            double arrival = currentTime + travel;

            int latest = patients[p].getEndTime();
            if (arrival > latest) {
                penalty += LATE_PENALTY_WEIGHT * (arrival - latest);
            }
            int earliest = patients[p].getStartTime();
            if (arrival < earliest) arrival = earliest;
            currentTime = arrival + patients[p].getCareTime();

            currentStrain += patients[p].getDemand();
            if (currentStrain > capacity) {
                penalty += CAPACITY_PENALTY_WEIGHT * (currentStrain - capacity);
            }
            prev = p;
        }

        double returnTravel = problem->getTravelTime(prev, DEPOT);
        totalTravel += returnTravel;
        double finishTime = currentTime + returnTravel;
        if (finishTime > returnTime) {
            penalty += LATE_PENALTY_WEIGHT * (finishTime - returnTime);
        }
    }

    if (activeRouteCount > maxNurses) {
        penalty += CAPACITY_PENALTY_WEIGHT * (activeRouteCount - maxNurses);
    }

    return totalTravel + penalty * penaltyMultiplier;
}

FitnessResult GA::evaluateDetailed(const Individual& ind) const {
    FitnessResult result = {0.0, 0.0, true};
    const auto& genes = ind.getGenes();
    if (genes.empty()) {
        result.rawTravelTime = numeric_limits<double>::max();
        result.isValid = false;
        return result;
    }

    const auto& patients = problem->getPatients();
    int capacity = problem->getCapacity();
    int returnTime = problem->getReturnTime();
    int maxNurses = problem->getNumberOfNurses();

    double totalTravel = 0;
    double penalty = 0;
    int prev = DEPOT;
    double currentTime = 0;
    int currentStrain = 0;
    int routeCount = 0;

    for (int gene : genes) {
        if (gene == ROUTE_SEPARATOR) {
            double returnTravel = problem->getTravelTime(prev, DEPOT);
            totalTravel += returnTravel;
            if (prev != DEPOT) {
                double finishTime = currentTime + returnTravel;
                if (finishTime > returnTime) {
                    penalty += LATE_PENALTY_WEIGHT * (finishTime - returnTime);
                    result.isValid = false;
                }
            }
            prev = DEPOT;
            currentTime = 0;
            currentStrain = 0;
            routeCount++;
            continue;
        }
        if (gene < 0 || gene >= static_cast<int>(patients.size())) continue;

        int patient = gene;
        double travel = problem->getTravelTime(prev, patient);
        double arrival = currentTime + travel;
        int demand = patients[patient].getDemand();
        int careTime = patients[patient].getCareTime();
        int earliest = patients[patient].getStartTime();
        int latest = patients[patient].getEndTime();

        totalTravel += travel;

        if (arrival > latest) {
            penalty += LATE_PENALTY_WEIGHT * (arrival - latest);
            result.isValid = false;
        }
        if (arrival < earliest) arrival = static_cast<double>(earliest);

        currentTime = arrival + careTime;
        currentStrain += demand;

        if (currentStrain > capacity) {
            penalty += CAPACITY_PENALTY_WEIGHT * (currentStrain - capacity);
            result.isValid = false;
        }
        prev = patient;
    }

    double lastReturnTravel = problem->getTravelTime(prev, DEPOT);
    totalTravel += lastReturnTravel;
    routeCount++;

    if (prev != DEPOT) {
        double finishTime = currentTime + lastReturnTravel;
        if (finishTime > returnTime) {
            penalty += LATE_PENALTY_WEIGHT * (finishTime - returnTime);
            result.isValid = false;
        }
    }
    if (routeCount > maxNurses) {
        penalty += CAPACITY_PENALTY_WEIGHT * (routeCount - maxNurses);
        result.isValid = false;
    }

    result.rawTravelTime = totalTravel;
    result.penalty = penalty;
    return result;
}

FitnessResult GA::evaluateFitnessDetailed(const Individual& ind) const {
    return evaluateDetailed(ind);
}

double GA::evaluateFitness(const Individual& ind) const {
    FitnessResult r = evaluateDetailed(ind);
    return r.rawTravelTime + r.penalty * penaltyMultiplier;
}

void GA::evaluatePopulation() {
    int populationCount = static_cast<int>(population.size());
    int feasibleCount = 0;

    for (int i = 0; i < populationCount; ++i) {
        FitnessResult result = evaluateDetailed(population[i]);
        population[i].setFitness(result.rawTravelTime + result.penalty * penaltyMultiplier);
        if (result.isValid) feasibleCount++;
    }

    double feasibilityRatio = static_cast<double>(feasibleCount) / std::max(populationCount, 1);
    if (feasibilityRatio > FEASIBILITY_UPPER_BOUND) {
        penaltyMultiplier *= PENALTY_DECREASE_RATE;
    } else if (feasibilityRatio < FEASIBILITY_LOWER_BOUND) {
        penaltyMultiplier *= PENALTY_INCREASE_RATE;
    }
    penaltyMultiplier = std::max(MIN_PENALTY_MULTIPLIER, std::min(MAX_PENALTY_MULTIPLIER, penaltyMultiplier));
}

const std::vector<Individual>& GA::getPopulation() const {
    return population;
}

std::vector<Individual>& GA::getPopulation() {
    return population;
}

Individual GA::getBest() const {
    if (population.empty()) return Individual(0, 0.0);
    auto best = min_element(population.begin(), population.end(),
        [](const Individual& a, const Individual& b) {
            return a.getFitness() < b.getFitness();
        });
    return *best;
}

int GA::getPopulationSize() const { return populationSize; }
double GA::getMutationRate() const { return mutationRate; }
void GA::setMutationRate(double r) { mutationRate = r; }
int GA::getEliteCount() const { return eliteCount; }
void GA::setEliteCount(int e) { eliteCount = e; }
double GA::getPenaltyMultiplier() const { return penaltyMultiplier; }
