#include "Similarity.hpp"
#include <vector>
#include <unordered_map>

using namespace std;

double cosineSimilarity(const vector<int>& a, const vector<int>& b) {
    double dot = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        dot   += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    if (normA == 0.0 || normB == 0.0) return 0.0;
    return dot / (sqrt(normA) * sqrt(normB));
}

double kendallTauSimilarity(const vector<int>& a, const vector<int>& b) {
    // Extract non-zero values with their original positions
    vector<pair<int, int>> elemsA, elemsB; // (value, position)
    
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != 0) elemsA.push_back({a[i], i});
        if (b[i] != 0) elemsB.push_back({b[i], i});
    }
    
    // Create position maps: value -> order in sequence
    unordered_map<int, int> posA, posB;
    for (size_t i = 0; i < elemsA.size(); i++) {
        posA[elemsA[i].first] = i;
    }
    for (size_t i = 0; i < elemsB.size(); i++) {
        posB[elemsB[i].first] = i;
    }
    
    // Find common elements
    vector<int> common;
    for (const auto& [val, _] : posA) {
        if (posB.count(val)) {
            common.push_back(val);
        }
    }
    
    if (common.size() < 2) {
        return (common.size() == elemsA.size() && common.size() == elemsB.size()) ? 1.0 : 0.0;
    }
    
    // Count concordant and discordant pairs
    int concordant = 0, discordant = 0;
    
    for (size_t i = 0; i < common.size(); i++) {
        for (size_t j = i + 1; j < common.size(); j++) {
            int val1 = common[i], val2 = common[j];
            
            // Check if ordering is same in both sequences
            bool orderA = posA[val1] < posA[val2];
            bool orderB = posB[val1] < posB[val2];
            
            if (orderA == orderB) {
                concordant++;
            } else {
                discordant++;
            }
        }
    }
    
    int totalPairs = concordant + discordant;
    if (totalPairs == 0) return 1.0;
    
    // Kendall Tau-a coefficient
    return (double)(concordant - discordant) / totalPairs;
}

double spearmanFootrule(const vector<int>& a, const vector<int>& b) {
    // Extract non-zero values
    vector<int> valsA, valsB;
    for (size_t i = 0; i < a.size(); i++) {
        if (a[i] != 0) valsA.push_back(a[i]);
        if (b[i] != 0) valsB.push_back(b[i]);
    }
    
    if (valsA.empty() || valsB.empty()) return 0.0;
    
    // Create rank maps: value -> position in sequence
    unordered_map<int, int> rankA, rankB;
    for (size_t i = 0; i < valsA.size(); i++) rankA[valsA[i]] = i;
    for (size_t i = 0; i < valsB.size(); i++) rankB[valsB[i]] = i;
    
    // Sum absolute differences in ranks for common elements
    int sumDiff = 0;
    int commonCount = 0;
    for (const auto& [val, rA] : rankA) {
        if (rankB.count(val)) {
            sumDiff += abs(rA - rankB[val]);
            commonCount++;
        }
    }
    
    if (commonCount == 0) return 0.0;
    
    // Normalize: max possible sum is when sequence is completely reversed
    // For n elements: 0 vs (n-1), 1 vs (n-2), ... = n*(n-1)/2
    int maxDiff = commonCount * (commonCount - 1) / 2;
    if (maxDiff == 0) return 1.0; // Only 1 common element
    
    return 1.0 - (double)sumDiff / maxDiff;
}
