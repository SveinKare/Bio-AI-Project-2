#include "individual.h"
#include <unordered_set>
#include <algorithm>

using namespace std;

std::mt19937 gen(1234);

Individual::Individual() : fitness(0.0) {}

Individual::Individual(int n, double f) : fitness(f), genes(n) {}

double Individual::getFitness() const {
    return fitness;
}

void Individual::setFitness(double f) {
    fitness = f;
}

const std::vector<int>& Individual::getGenes() const {
    return genes;
}

void Individual::setGenes(const std::vector<int>& g) {
    genes = g;
}

void Individual::setGenes(std::vector<int>&& g) {
    genes = std::move(g);
}

void Individual::swapPatients(size_t idx1, size_t idx2) {
    if (idx1 < genes.size() && idx2 < genes.size()) {
        std::swap(genes[idx1], genes[idx2]);
    }
}

static std::vector<std::vector<int>> genesToRoutes(const std::vector<int>& g) {
    std::vector<std::vector<int>> routes;
    std::vector<int> current;
    for (int x : g) {
        if (x == -1) {
            if (!current.empty()) {
                routes.push_back(std::move(current));
                current.clear();
            }
        } else if (x > 0) {
            current.push_back(x);
        }
    }
    if (!current.empty()) routes.push_back(std::move(current));
    return routes;
}

static std::vector<int> routesToGenes(const std::vector<std::vector<int>>& routes) {
    std::vector<int> g;
    for (size_t i = 0; i < routes.size(); ++i) {
        if (i > 0) g.push_back(-1);
        for (int p : routes[i]) g.push_back(p);
    }
    return g;
}

void Individual::mutation(double mutationRate) {
    mutation(mutationRate, gen);
}

void Individual::mutation(double mutationRate, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    if (u(rng) >= mutationRate) return;

    if (u(rng) < 0.5) {
        relocateMutation(rng);
    } else {
        swapMutation(rng);
    }
}

void Individual::relocateMutation() {
    relocateMutation(gen);
}

void Individual::relocateMutation(std::mt19937& rng) {
    auto routes = genesToRoutes(genes);
    if (routes.size() < 2) return;

    std::uniform_int_distribution<size_t> routeDist(0, routes.size() - 1);
    size_t fromRoute = routeDist(rng);
    if (routes[fromRoute].empty()) return;

    std::uniform_int_distribution<size_t> posDist(0, routes[fromRoute].size() - 1);
    size_t patientIdx = posDist(rng);
    int patient = routes[fromRoute][patientIdx];
    routes[fromRoute].erase(routes[fromRoute].begin() + static_cast<std::ptrdiff_t>(patientIdx));

    size_t toRoute = routeDist(rng);
    while (toRoute == fromRoute) toRoute = routeDist(rng);

    std::uniform_real_distribution<double> u(0.0, 1.0);
    size_t insertPos = static_cast<size_t>(u(rng) * (routes[toRoute].size() + 1));
    if (insertPos > routes[toRoute].size()) insertPos = routes[toRoute].size();
    routes[toRoute].insert(routes[toRoute].begin() + static_cast<std::ptrdiff_t>(insertPos), patient);

    if (routes[fromRoute].empty()) {
        routes.erase(routes.begin() + static_cast<std::ptrdiff_t>(fromRoute));
    }

    genes = routesToGenes(routes);
}

void Individual::swapMutation() {
    swapMutation(gen);
}

void Individual::swapMutation(std::mt19937& rng) {
    std::vector<size_t> patientPos;
    for (size_t i = 0; i < genes.size(); ++i) {
        if (genes[i] > 0) patientPos.push_back(i);
    }
    if (patientPos.size() < 2) return;
    std::uniform_real_distribution<double> u(0.0, 1.0);
    size_t i = static_cast<size_t>(u(rng) * patientPos.size()) % patientPos.size();
    size_t j = static_cast<size_t>(u(rng) * patientPos.size()) % patientPos.size();
    if (i != j) std::swap(genes[patientPos[i]], genes[patientPos[j]]);
}

void Individual::reset() {
    genes.clear();
    fitness = 0.0;
}

Individual Individual::crossover(const Individual& parent1, const Individual& parent2) {
    const auto& g1 = parent1.getGenes();
    const auto& g2 = parent2.getGenes();
    size_t n = g1.size();

    if (n != g2.size() || n == 0) return Individual(0, 0.0);

    std::uniform_int_distribution<size_t> dist(0, n - 1);
    size_t start = dist(gen);
    size_t end = dist(gen);
    if (start > end) std::swap(start, end);

    std::vector<int> child(n, -1);
    std::unordered_set<int> used;

    for (size_t i = start; i <= end; ++i) {
        child[i] = g1[i];
        used.insert(g1[i]);
    }

    size_t childIdx = (end + 1) % n;
    size_t parentIdx = (end + 1) % n;
    size_t placed = 0;
    const size_t toPlace = n - (end - start + 1);

    while (placed < toPlace) {
        int val = g2[parentIdx];
        if (used.find(val) == used.end()) {
            child[childIdx] = val;
            used.insert(val);
            childIdx = (childIdx + 1) % n;
            ++placed;
        }
        parentIdx = (parentIdx + 1) % n;
    }

    Individual offspring(static_cast<int>(n), 0.0);
    offspring.setGenes(child);
    return offspring;
}
