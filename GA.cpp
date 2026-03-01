#include "GA.h"
#include "HomeCare.cpp"
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>
#include <map>

using namespace std;

static constexpr int DEPOT = 0;
static constexpr int ROUTE_SEPARATOR = -1;

static constexpr double LATE_PENALTY_WEIGHT = 10000.0;
static constexpr double CAPACITY_PENALTY_WEIGHT = 10000.0;

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
        } else if (gene > 0) {
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

void GA::repairRouteCount(Individual& ind) const {
    vector<vector<int>> routes = genomeToRoutes(ind.getGenes());
    int maxNurses = problem->getNumberOfNurses();
    while (static_cast<int>(routes.size()) > maxNurses) {
        size_t a = 0, b = 1;
        if (routes[1].size() < routes[0].size()) std::swap(a, b);
        for (size_t i = 2; i < routes.size(); ++i) {
            if (routes[i].size() < routes[a].size()) {
                b = a;
                a = i;
            } else if (routes[i].size() < routes[b].size()) {
                b = i;
            }
        }
        routes[a].insert(routes[a].end(), routes[b].begin(), routes[b].end());
        routes.erase(routes.begin() + static_cast<ptrdiff_t>(b));
    }
    ind.setGenes(routesToGenome(routes));
}

GA::GA(HomeCare* hc, int popSize, double mutRate, int elites, int tournSize,
       double sf)
    : problem(hc), populationSize(popSize), mutationRate(mutRate),
      eliteCount(elites), tournamentSize(tournSize),
      generationCount(0), crowdingScalingFactor(sf) {}

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

int GA::similarity(const Individual& a, const Individual& b) const {
    int totalPatients = problem->getNbrPatients();
    vector<int> routeA(totalPatients + 1, -1);
    vector<int> routeB(totalPatients + 1, -1);

    int routeIndex = 0;
    for (int gene : a.getGenes()) {
        if (gene == ROUTE_SEPARATOR) ++routeIndex;
        else if (gene > 0 && gene <= totalPatients) routeA[gene] = routeIndex;
    }
    routeIndex = 0;
    for (int gene : b.getGenes()) {
        if (gene == ROUTE_SEPARATOR) ++routeIndex;
        else if (gene > 0 && gene <= totalPatients) routeB[gene] = routeIndex;
    }

    
    map<pair<int,int>, int> cells;
    for (int p = 1; p <= totalPatients; ++p) {
        if (routeA[p] >= 0 && routeB[p] >= 0)
            cells[{routeA[p], routeB[p]}]++;
    }

    int sharedPairs = 0;
    for (auto& [key, count] : cells)
        sharedPairs += count * (count - 1) / 2;
    return sharedPairs;
}

void GA::runGeneration() {
    if (population.empty()) return;
    generationCount++;

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

    int childrenCount = populationSize - actualElites;
    vector<Individual> children(childrenCount);

    {
        mt19937 offspringRng(static_cast<unsigned>(
            2000000u + generationCount * 104729u));
        uniform_real_distribution<double> unitDist(0.0, 1.0);

        for (int c = 0; c < childrenCount; ++c) {
            int parentIdx1 = tournamentSelect(offspringRng);
            int parentIdx2 = tournamentSelect(offspringRng);
            while (parentIdx2 == parentIdx1) parentIdx2 = tournamentSelect(offspringRng);

            Individual child = crossoverFlat(population[parentIdx1], population[parentIdx2]);
            child.mutation(mutationRate, offspringRng);
            repairRouteCount(child);
            child.setFitness(evaluateFitness(child));

            int similarityToParent1 = similarity(child, population[parentIdx1]);
            int similarityToParent2 = similarity(child, population[parentIdx2]);
            int closerParentIdx = (similarityToParent1 >= similarityToParent2) ? parentIdx1 : parentIdx2;

            double cf = child.getFitness();
            double pf = population[closerParentIdx].getFitness();
            double pDet = (cf <= pf) ? 1.0 : 0.0;
            double denom = cf + pf;
            double pProb = (denom > 0) ? pf / denom : 0.5;
            double childProb = (1.0 - crowdingScalingFactor) * pDet
                             + crowdingScalingFactor * pProb;

            if (unitDist(offspringRng) < childProb) {
                children[c] = std::move(child);
            } else {
                children[c] = population[closerParentIdx];
            }
        }
    }

    vector<Individual> nextGeneration;
    nextGeneration.reserve(populationSize);
    for (auto& elite : elites) nextGeneration.push_back(std::move(elite));
    for (auto& child : children) nextGeneration.push_back(std::move(child));

    population = std::move(nextGeneration);
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
    vector<bool> assigned(static_cast<size_t>(n + 1), false);
    vector<vector<int>> routes;

    int prev = DEPOT;
    double currentTime = 0;
    int currentStrain = 0;
    vector<int> currentRoute;

    vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i + 1;

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

    for (int p = 1; p <= n; ++p) {
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
    for (int i = 0; i < n; ++i) indices[i] = i + 1;
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

void GA::initialize() {
    population.clear();
    generationCount = 0;
    int n = problem->getNbrPatients();
    if (n == 0) return;

    population.resize(populationSize);
    mt19937 local_rng(static_cast<unsigned>(3000000u));

    int greedyCount = std::max(1, populationSize / 4);
    for (int i = 0; i < greedyCount; ++i) {
        population[i] = createGreedyIndividual(local_rng);
        repairRouteCount(population[i]);
    }
    for (int i = greedyCount; i < populationSize; ++i) {
        population[i] = createRandomIndividual(local_rng);
        repairRouteCount(population[i]);
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
    vector<bool> assigned(static_cast<size_t>(nbrPatients + 1), false);

    for (const auto& sr : all) {
        bool ok = true;
        for (int p : sr.route) if (assigned[p]) { ok = false; break; }
        if (ok) {
            childRoutes.push_back(sr.route);
            for (int p : sr.route) assigned[p] = true;
        }
    }

    for (int p = 1; p <= nbrPatients; ++p) {
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

    return totalTravel + penalty;
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
        if (gene <= 0 || gene >= static_cast<int>(patients.size())) continue;

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
    return r.rawTravelTime + r.penalty;
}

void GA::evaluatePopulation() {
    for (size_t i = 0; i < population.size(); ++i) {
        population[i].setFitness(evaluateFitness(population[i]));
    }
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
