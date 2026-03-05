#include "RuinAndRepair.hpp"
#include "KMeans.hpp"
#include "individual.h"
#include <limits>
#include <random>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "Similarity.hpp"

using namespace std;

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
    int kElites,
    SimilarityFunc similarityFunc,
    unsigned int seed)
    : homeCare(homeCare), 
    popSize(popSize), 
    epsilon(epsilon), 
    kParents(kParents), 
    generations(generations), 
    penalty(penalty), 
    crossoverRate(crossoverRate), 
    mutationRate(mutationRate),
    scalingFactor(scalingFactor),
    kElites(kElites),
    similarityFunc(similarityFunc),
    rng(seed)
{}

void RuinAndRepair::setPopulation(vector<Individual>& individuals) {
  this->population = individuals;
  for (auto& ind : this->population) {
    auto fitness = homeCare.calculateFitness(ind.getGenes(), this->penalty);
    ind.setFitness(fitness.first + fitness.second);
    ind.setPenalty(fitness.second);
  }
}

Individual RuinAndRepair::getBestSolution() {
  double minFitness = numeric_limits<double>::max();
  Individual best;
  best.setFitness(minFitness);

  for (const auto& ind : population) {
    if (ind.getFitness() > minFitness) continue;

    if (ind.getPenalty() == 0.0) {
      // Fitness has no penalty, which means it's a legal solution
      minFitness = ind.getFitness();
      best = ind;
    }
  }

  return best;
}

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
  // Create list of all patients
  vector<int> patients;
  for (int i = 1; i <= homeCare.getNbrPatients(); i++) {
    patients.push_back(i);
  }
  
  // Shuffle patients randomly
  shuffle(patients.begin(), patients.end(), rng);
  
  // Distribute patients round-robin to nurses
  vector<vector<int>> routes(homeCare.getNumberOfNurses());
  
  // Initialize all routes with depot
  for (size_t i = 0; i < routes.size(); i++) {
    routes[i].push_back(0);
  }
  
  // Assign patients round-robin
  size_t currNurse = 0;
  for (int patient : patients) {
    routes[currNurse].push_back(patient);
    currNurse = (currNurse + 1) % homeCare.getNumberOfNurses();
  }
  
  // Build gene from routes
  vector<int> gene;
  for (auto& route : routes) {
    gene.insert(gene.end(), route.begin(), route.end());
  }
  gene.push_back(0);  // Final depot

  uniform_real_distribution<double> scalingDist(0.0, scalingFactor);
  
  // Create individual
  auto individual = Individual();
  individual.setGenes(gene);
  auto fitness = homeCare.calculateFitness(gene, this->penalty);
  individual.setFitness(fitness.first + fitness.second);
  individual.setPenalty(fitness.second);
  individual.setScalingFactor(scalingDist(rng));
  
  return individual;
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
  vector<Individual> elites;
  for (auto& ind : population) {
    if (ind.getPenalty() == 0.0) {
      elites.push_back(ind);
    }
  }
  if (elites.size() < kElites) {
    // If there are not enough valid solutions, we the best non-valid ones
    vector<Individual> sorted = this->population;
    sort(sorted.begin(), sorted.end(), [](const Individual& a, const Individual& b) {
        return a.getFitness() < b.getFitness(); 
        });
    elites.insert(elites.end(), sorted.begin(), sorted.begin() + (kElites - elites.size()));
  }

  sort(elites.begin(), elites.end(), [](const Individual& a, const Individual& b) {
      return a.getFitness() < b.getFitness(); 
      });

  if (elites.size() < kElites) throw runtime_error("Too few elites");

  return vector<Individual>(elites.begin(), elites.begin()+kElites);
}

Individual RuinAndRepair::tournamentParentSelection(vector<Individual>::iterator begin, vector<Individual>::iterator end) {
  if (begin == end) throw runtime_error("Parent selection attempted with no candidates");
  Individual* best = nullptr;
  double min = numeric_limits<double>::max();
  for (auto it = begin; it != end; it++) {
    double fitness = it->getFitness();
    if (fitness < min) {  
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

  double inheritedScalingFactor = (parents[0].getScalingFactor() + parents[1].getScalingFactor()) / 2;
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

    auto fitness = homeCare.calculateFitness(gene, this->penalty);
    c.setGenes(gene);
    c.setFitness(fitness.first + fitness.second);
    c.setPenalty(fitness.second);
    c.setScalingFactor(inheritedScalingFactor);
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
  uniform_real_distribution<double> choice(0.0, 1.0);
  double r = choice(rng);
  
  if (r < 0.3) {
    twoOptMutation(individual);
  } else if (r < 0.7) {
    relocateMutation(individual);
  } else {
    exchangeMutation(individual);
  }

  normal_distribution<double> gaussian(0.0, this->scalingFactor * 0.1);
  double newScalingFactor = min(this->scalingFactor, (individual.getScalingFactor() + gaussian(rng)) );

  auto fitness = homeCare.calculateFitness(individual.getGenes(), this->penalty);
  individual.setFitness(fitness.first + fitness.second);
  individual.setPenalty(fitness.second);
  individual.setScalingFactor(max(0.0, newScalingFactor));
}

void RuinAndRepair::twoOptMutation(Individual& individual) {
  vector<int> gene = individual.getGenes();
  
  // Find route boundaries
  vector<int> depots;
  for (size_t i = 0; i < gene.size(); i++) {
    if (gene[i] == 0) depots.push_back(i);
  }
  
  if (depots.size() <= 2) return;
  
  // Pick random route
  uniform_int_distribution<int> routeDist(0, depots.size() - 2);
  int routeIdx = routeDist(rng);
  int start = depots[routeIdx];
  int end = depots[routeIdx + 1];
  
  if (end - start <= 2) return;  // Need at least 2 patients
  
  // Pick two positions
  uniform_int_distribution<int> posDist(start + 1, end - 1);
  int i = posDist(rng);
  int j = posDist(rng);
  
  if (i > j) swap(i, j);
  if (i == j) return;
  
  // Reverse segment
  reverse(gene.begin() + i, gene.begin() + j + 1);
  
  individual.setGenes(gene);
}

void RuinAndRepair::relocateMutation(Individual& individual) {
  vector<int> gene = individual.getGenes();
  
  // Find all patient positions
  vector<int> patientPos;
  for (size_t i = 0; i < gene.size(); i++) {
    if (gene[i] != 0) patientPos.push_back(i);
  }
  
  if (patientPos.empty()) return;
  
  // Pick random patient
  uniform_int_distribution<int> patDist(0, patientPos.size() - 1);
  int removeIdx = patientPos[patDist(rng)];
  int patient = gene[removeIdx];
  
  // Remove patient
  gene.erase(gene.begin() + removeIdx);
  
  // Insert at random position (not at depot)
  vector<int> validInsertPos;
  for (size_t i = 1; i < gene.size(); i++) {
    if (gene[i-1] != 0 || gene[i] != 0) {  // Not between two depots
      validInsertPos.push_back(i);
    }
  }
  
  if (validInsertPos.empty()) return;
  
  uniform_int_distribution<int> insDist(0, validInsertPos.size() - 1);
  int insertIdx = validInsertPos[insDist(rng)];
  
  gene.insert(gene.begin() + insertIdx, patient);
  
  individual.setGenes(gene);
}

void RuinAndRepair::exchangeMutation(Individual& individual) {
  vector<int> gene = individual.getGenes();
  
  // Find all patient positions
  vector<int> patientPos;
  for (size_t i = 0; i < gene.size(); i++) {
    if (gene[i] != 0) patientPos.push_back(i);
  }
  
  if (patientPos.size() < 2) return;
  
  // Pick two random patients
  uniform_int_distribution<int> patDist(0, patientPos.size() - 1);
  int pos1 = patientPos[patDist(rng)];
  int pos2 = patientPos[patDist(rng)];
  
  while (pos1 == pos2 && patientPos.size() > 1) {
    pos2 = patientPos[patDist(rng)];
  }
  
  // Swap
  swap(gene[pos1], gene[pos2]);
  
  individual.setGenes(gene);
}

void RuinAndRepair::generalizedCrowding(vector<Individual>& parents, vector<Individual>& children, vector<Individual>& survivors) {
  uniform_real_distribution<double> randEvent(0.0, 1.0);
  // MINIMIZE FITNESS
  double simA = similarityFunc(parents[0].getGenes(), children[0].getGenes()) + similarityFunc(parents[1].getGenes(), children[1].getGenes());
  double simB = similarityFunc(parents[1].getGenes(), children[0].getGenes()) + similarityFunc(parents[0].getGenes(), children[1].getGenes());

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

  double sf1 = parents[0].getFitness() < o1.getFitness() ? parents[0].getScalingFactor() : o1.getScalingFactor();
  double sf2 = parents[1].getFitness() < o2.getFitness() ? parents[1].getScalingFactor() : o2.getScalingFactor();

  // Inverted formula due to minimizing fitness
  auto replacementProb = [&](double fitnessChild, double fitnessParent, double sf) {
    return fitnessParent / (fitnessParent + sf * fitnessChild);
  };

  if (randEvent(rng) < replacementProb(o1.getFitness(), parents[0].getFitness(), sf1)) {
    survivors.push_back(o1);
  } else {
    survivors.push_back(parents[0]);
  }

  if (randEvent(rng) < replacementProb(o2.getFitness(), parents[1].getFitness(), sf2)) {
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
      }
      if (randEvent(rng) < this->mutationRate) {
        this->mutate(children[1]);
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

void RuinAndRepair::runGenerations(int generations) {
  uniform_int_distribution<int> randIndex(0, this->popSize-1);
  uniform_real_distribution<double> randEvent(0.0, 1.0);
  for (int g = 0; g < generations; g++) {
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
      }
      if (randEvent(rng) < this->mutationRate) {
        this->mutate(children[1]);
      }

      // Crowding (group by cosine similarity)
      vector<Individual> survivors;
      this->generalizedCrowding(parents, children, survivors);

      newPop.insert(newPop.end(), survivors.begin(), survivors.end());
      newPopSize += 2;
    }
    this->population = std::move(newPop);
    cout << "Generation: " << g << endl;
    //this->printPopulationStats();
  }
}

vector<Individual> RuinAndRepair::getBestIndividuals(int k) const {
  auto sorted = population;
  sort(sorted.begin(), sorted.end(), [](const Individual& a, const Individual& b) {
      return a.getFitness() < b.getFitness();
      });
  return vector<Individual>(sorted.begin(), sorted.begin() + k);
}

void RuinAndRepair::injectIndividuals(vector<Individual>& immigrants) {
    sort(population.begin(), population.end(), [](const Individual& a, const Individual& b) {
        return a.getFitness() > b.getFitness();  
    });
    for (int i = 0; i < immigrants.size(); i++) {
        population[i] = immigrants[i];  
    }
}
