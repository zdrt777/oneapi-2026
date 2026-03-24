#include "permutations_cxx.h"

void Permutations(dictionary_t& dictionary) {
    dictionary_t groups;

    for (auto const& item : dictionary) {
        std::string sorted_key = item.first;
        std::sort(sorted_key.begin(), sorted_key.end());
        groups[sorted_key].push_back(item.first);
    }

    for (auto& item : dictionary) {
        const std::string& original_key = item.first;
        
        std::string sorted_key = item.first;
        std::sort(sorted_key.begin(), sorted_key.end());

        const std::vector<std::string>& group = groups[sorted_key];
        std::vector<std::string>& result_vec = item.second;

        for (const std::string& candidate : group) {
            if (candidate != original_key) {
                result_vec.push_back(candidate);
            }
        }

        std::sort(result_vec.begin(), result_vec.end(), std::greater<std::string>());
    }
}