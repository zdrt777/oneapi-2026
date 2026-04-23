#include "permutations_cxx.h"
#include <algorithm>

void Permutations(dictionary_t& dictionary) {
    std::map<std::string, std::vector<std::string>> groups;
    for (const auto& entry : dictionary) {
        const std::string& key = entry.first;
        std::string canonical = key;
        std::sort(canonical.begin(), canonical.end());
        groups[canonical].push_back(key);
    }
    for (auto& entry : dictionary) {
        const std::string& key = entry.first;
        std::string canonical = key;
        std::sort(canonical.begin(), canonical.end());

        const auto& group = groups[canonical];
        std::vector<std::string> permutations;
        permutations.reserve(group.size() - 1);

        for (const std::string& s : group) {
            if (s != key) {
                permutations.push_back(s);
            }
        }
        std::sort(permutations.begin(), permutations.end(),
                  std::greater<std::string>());

        entry.second = std::move(permutations);
    }
}