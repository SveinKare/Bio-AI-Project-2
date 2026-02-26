#include "RuinAndRepair.hpp"
#include "KMeans.hpp"
#include "individual.h"
#include <random>
#include <iostream>
#include <vector>
#include "Similarity.hpp"

using namespace std;

mt19937 rng(random_device{}());

RuinAndRepair::RuinAndRepair(
    HomeCare& homeCare, 
    int popSize, 
    double epsilon, 
    int kParents, 
    int generations, 
    double penalty, 
    double crossoverRate, 
    double mutationRate,
    double scalingFactor,
    int kElites)
    : homeCare(homeCare), 
    popSize(popSize), 
    epsilon(epsilon), 
    kParents(kParents), 
    generations(generations), 
    penalty(penalty), 
    crossoverRate(crossoverRate), 
    mutationRate(mutationRate),
    scalingFactor(scalingFactor),
    kElites(kElites)
{}

double RuinAndRepair::getMinFitness() {
  double minFitness = numeric_limits<double>::max();

  for (const auto& ind : population) {
    minFitness = min(minFitness, ind.getFitness());
  }
  return minFitness;
}

void RuinAndRepair::printPopulationStats() {
  double minFitness = numeric_limits<double>::max();
  double maxFitness = numeric_limits<double>::lowest();

  for (const auto& ind : population) {
    minFitness = min(minFitness, ind.getFitness());
    maxFitness = max(maxFitness, ind.getFitness());
  }

  cout << "Min fitness: " << minFitness << " | Max fitness: " << maxFitness << endl;
}

Individual RuinAndRepair::randomIndividual(vector<vector<int>>& clusters) {
  uniform_real_distribution<double> chooseRandom(0.0, 1.0);
  uniform_int_distribution<int> randomCluster(0, homeCare.getNumberOfNurses() - 1);
  size_t currNurse = 0;
  int assignedPatients = 0;

  // Init routes, all routes start at depot
  vector<vector<int>> routes(homeCare.getNumberOfNurses());
  vector<bool> assigned(homeCare.getNbrPatients(), false);
  for (size_t i = 0; i < routes.size(); i++) {
    routes[i].push_back(0);
  }

  while (assignedPatients < homeCare.getNbrPatients()) {
    size_t clusterIdx = currNurse;

    // Choose a patient from a random route with prob. epsilon
    if (chooseRandom(rng) < this->epsilon) {
      clusterIdx = randomCluster(rng);
    }

    // Find the nearest neighbor
    int bestPatient = -1;
    double shortestTravelTime = INFINITY;
    int lastPatient = routes[currNurse].back();
    for (size_t i = 0; i < clusters[clusterIdx].size(); i++) {
      int patientIdx = clusters[clusterIdx][i];
      if (assigned[patientIdx-1]) continue;
      double travelTime = homeCare.getTravelTime(lastPatient, patientIdx);
      if (travelTime < shortestTravelTime) {
        shortestTravelTime = travelTime;
        bestPatient = patientIdx;
      }
    }

    // Add to nurses route
    if (bestPatient > 0) {
      routes[currNurse].push_back(bestPatient);
      assigned[bestPatient-1] = true;
      assignedPatients++;
    }

    currNurse = (currNurse + 1) % homeCare.getNumberOfNurses();
  }
  vector<int> gene;
  for (auto&v : routes) {
    gene.insert(gene.end(), v.begin(), v.end());
  }
  gene.push_back(0);
  auto i = Individual();
  i.setFitness(homeCare.calculateFitness(gene, this->penalty));
  i.setGenes(gene);
  return i;
}

void RuinAndRepair::initPopulation() {
  vector<Point> points;

  // CLustering using kmeans++
  for (int i = 1; i <= homeCare.getNbrPatients(); i++) {
    Patient p = homeCare.getPatients()[i];
    points.push_back({static_cast<double>(p.getXCoord()), static_cast<double>(p.getYCoord()), i});
  }

  KMeans kmeans(homeCare.getNumberOfNurses());
  vector<vector<int>> clusters = kmeans.fit(points);

  for (int i = 0; i < this->popSize; i++) {
    this->population.push_back(randomIndividual(clusters));
  }
}

vector<Individual> RuinAndRepair::eliteSelection() {
  vector<Individual> sorted = this->population;
  sort(sorted.begin(), sorted.end(), [](const Individual& a, const Individual& b) {
      return a.getFitness() < b.getFitness(); 
      });
  return vector<Individual>(sorted.begin(), sorted.begin() + kElites);
}

Individual RuinAndRepair::tournamentParentSelection(vector<Individual>::iterator begin, vector<Individual>::iterator end) {
  if (begin == end) throw runtime_error("Parent selection attempted with no candidates");
  Individual* best = nullptr;
  double min = numeric_limits<double>::max();
  for (auto it = begin; it != end; it++) {
    double fitness = it->getFitness();
    if (fitness < min) {  // ← minimizing
      min = fitness;
      best = &(*it);
    }
  }
  return *best;
}

void RuinAndRepair::orderCrossover(vector<Individual>& parents, vector<Individual>& children) {
  if (parents.empty() || !children.empty()) throw runtime_error("Crossover attempted with empty parents or non-empty children");
  int geneLength = parents[0].getGenes().size();
  uniform_int_distribution<int> points(1, geneLength-2); // We add on start and end depot at the end
  int first = points(rng);
  int second;
  do {
    second = points(rng);
  } while(first == second); // Ensures a and b are different

  // Sort
  if (first > second) {
    int k = second;
    second = first;
    first = k;
  }

  int allowedZeros = homeCare.getNumberOfNurses()-1;

  for (int i = 0; i < 2; i++) {
    // p1 = parents[i]
    // p2 = parents[(i+1)%2]
    Individual c;

    vector<int> gene(geneLength);
    vector<bool> assigned(homeCare.getNbrPatients(), false);
    int zerosPlaced = 0;
    int valuesPlaced = 0;

    for (int j = first; j <= second; j++) {
      int curr = parents[i].getGenes()[j];
      gene[j] = curr;
      if (curr == 0) {
        zerosPlaced++;
      } else {
        assigned[curr-1] = true;
      }
      valuesPlaced++;
    }

    int placeNext = (second%(geneLength-2)) + 1; // Pointer for the child
    for (int j = (second%(geneLength-2)) + 1; valuesPlaced < geneLength-2; j = (j%(geneLength-2)) + 1) {
      int curr = parents[(i+1)%2].getGenes()[j];
      if (curr == 0 && zerosPlaced < allowedZeros) {
        gene[placeNext] = curr;
        placeNext = (placeNext%(geneLength-2)) + 1;
        zerosPlaced++;
        valuesPlaced++;
      } else if (curr != 0 && !assigned[curr-1]) {
        gene[placeNext] = curr;
        placeNext = (placeNext%(geneLength-2)) + 1;
        assigned[curr-1] = true;
        valuesPlaced++;
      }
    }

    int fitness = homeCare.calculateFitness(gene, this->penalty);
    c.setGenes(gene);
    c.setFitness(fitness);
    children.push_back(c);
  }
}

void printIndividual(const Individual& ind, const std::string& label) {
  std::cout << label << ": [ ";
  for (int g : ind.getGenes()) {
    std::cout << g << " ";
  }
  std::cout << "] fitness=" << ind.getFitness() << "\n";
}

void RuinAndRepair::test() {
  Individual p1, p2;
  p1.setGenes({0, 1, 0, 2, 3, 0, 4, 5, 0});
  p2.setGenes({0, 3, 0, 1, 4, 0, 5, 2, 0});

  std::vector<Individual> parents = {p1, p2};
  std::vector<Individual> children;

  printIndividual(p1, "Parent 1");
  printIndividual(p2, "Parent 2");
  std::cout << "\n";

  this->orderCrossover(parents, children);

  for (int i = 0; i < children.size(); i++) {
    printIndividual(children[i], "Child " + std::to_string(i+1));
  }
}

void RuinAndRepair::mutate(Individual& individual) {
  vector<int> gene = individual.getGenes();
  double ruinFraction = 0.05;

  // Only operate on the inner part of the gene (excluding first and last depot)
  vector<int> inner(gene.begin() + 1, gene.end() - 1);
  int innerSize = (int)inner.size();
  int ruinSize = max(1, (int)(innerSize * ruinFraction));

  uniform_int_distribution<int> startDist(0, innerSize - ruinSize);
  int startIdx = startDist(rng);

  // --- RUIN: extract segment from inner ---
  vector<int> ruined(inner.begin() + startIdx, inner.begin() + startIdx + ruinSize);
  vector<int> remaining;
  remaining.insert(remaining.end(), inner.begin(), inner.begin() + startIdx);
  remaining.insert(remaining.end(), inner.begin() + startIdx + ruinSize, inner.end());

  // --- REPAIR: split ruined into patients and depots ---
  vector<int> ruinedPatients, ruinedDepots;
  for (int val : ruined) {
    if (val == 0) ruinedDepots.push_back(val);
    else ruinedPatients.push_back(val);
  }

  // Reinsert patients using cheapest insertion
  shuffle(ruinedPatients.begin(), ruinedPatients.end(), rng);
  for (int val : ruinedPatients) {
    double bestCost = numeric_limits<double>::max();
    int bestPos = (int)remaining.size();
    for (int i = 1; i < (int)remaining.size(); i++) {
      int prev = remaining[i-1];
      int next = remaining[i];
      double insertionCost = homeCare.getTravelTime(prev, val)
        + homeCare.getTravelTime(val, next)
        - homeCare.getTravelTime(prev, next);
      if (insertionCost < bestCost) {
        bestCost = insertionCost;
        bestPos = i;
      }
    }
    remaining.insert(remaining.begin() + bestPos, val);
  }

  // Reinsert depots randomly
  for (int dep : ruinedDepots) {
    if (remaining.size() <= 1) {
      remaining.push_back(dep);
      continue;
    }
    uniform_int_distribution<int> depotDist(1, (int)remaining.size() - 1);
    remaining.insert(remaining.begin() + depotDist(rng), dep);
  }

  // Rebuild full gene with preserved start and end depots
  vector<int> newGene;
  newGene.push_back(0);
  newGene.insert(newGene.end(), remaining.begin(), remaining.end());
  newGene.push_back(0);

  individual.setGenes(newGene);
  individual.setFitness(homeCare.calculateFitness(newGene, this->penalty));
}

void RuinAndRepair::generalizedCrowding(vector<Individual>& parents, vector<Individual>& children, vector<Individual>& survivors) {
  uniform_real_distribution<double> randEvent(0.0, 1.0);
  // MINIMIZE FITNESS
  double simA = cosineSimilarity(parents[0].getGenes(), children[0].getGenes()) + cosineSimilarity(parents[1].getGenes(), children[1].getGenes());
  double simB = cosineSimilarity(parents[1].getGenes(), children[0].getGenes()) + cosineSimilarity(parents[0].getGenes(), children[1].getGenes());

  // With cosine similarity we want to maximize the value
  Individual o1;
  Individual o2;
  if (simA >= simB) {
    // (p1, c1) and (p2, c2)
    o1 = children[0];
    o2 = children[1];
  } else {
    o1 = children[1];
    o2 = children[0];
  }

  // Inverted formula due to minimizing fitness
  auto replacementProb = [&](double fitnessChild, double fitnessParent) {
    return fitnessParent / (fitnessParent + scalingFactor * fitnessChild);
  };

  if (randEvent(rng) < replacementProb(o1.getFitness(), parents[0].getFitness())) {
    survivors.push_back(o1);
  } else {
    survivors.push_back(parents[0]);
  }

  if (randEvent(rng) < replacementProb(o2.getFitness(), parents[1].getFitness())) {
    survivors.push_back(o2);
  } else {
    survivors.push_back(parents[1]);
  }
}

void RuinAndRepair::run() {
  this->initPopulation();
  uniform_int_distribution<int> randIndex(0, this->popSize-1);
  uniform_real_distribution<double> randEvent(0.0, 1.0);

  for (int g = 0; g < this->generations; g++) {
    int newPopSize = kElites;
    vector<Individual> newPop;
    auto elites = this->eliteSelection();
    newPop.insert(newPop.end(), elites.begin(), elites.end());

    while (newPopSize < popSize) {
      // Select parents
      vector<Individual> candidates;
      for (int i = 0; i < kParents*2; i++) {
        candidates.push_back(population[randIndex(rng)]);
      }

      Individual first = tournamentParentSelection(candidates.begin(), candidates.begin()+kParents);
      Individual second = tournamentParentSelection(candidates.begin()+kParents, candidates.end());


      vector<Individual> children;
      vector<Individual> parents = {first,second};
      if (randEvent(rng) < this->crossoverRate) {
        // Recombine
        this->orderCrossover(parents, children);
      } else {
        // Copy the children
        children = parents;
      }

      // Mutate
      if (randEvent(rng) < this->mutationRate) {
        this->mutate(children[0]);
        this->mutate(children[1]);
        children[0].setFitness(homeCare.calculateFitness(children[0].getGenes(), this->penalty));
        children[1].setFitness(homeCare.calculateFitness(children[1].getGenes(), this->penalty));
      }

      // Crowding (group by cosine similarity)
      vector<Individual> survivors;
      this->generalizedCrowding(parents, children, survivors);

      newPop.insert(newPop.end(), survivors.begin(), survivors.end());
      newPopSize += 2;
    }
    this->population = std::move(newPop);
    this->printPopulationStats();
  }
}
