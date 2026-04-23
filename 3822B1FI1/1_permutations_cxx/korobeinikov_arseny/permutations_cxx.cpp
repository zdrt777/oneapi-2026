#include "permutations_cxx.h"

#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t& dictionary) {
    std::unordered_map<std::string, std::vector<std::string>> classes;

    for (const auto& item : dictionary) {
        std::string sorted_word = item.first;
        std::sort(sorted_word.begin(), sorted_word.end());
        classes[sorted_word].push_back(item.first);
    }

    for (auto& group : classes) {
        std::vector<std::string>& words = group.second;
        std::sort(words.begin(), words.end(), std::greater<std::string>());

        for (const std::string& word : words) {
            std::vector<std::string>& result = dictionary[word];
            result.clear();

            for (const std::string& candidate : words) {
                if (candidate != word) {
                    result.push_back(candidate);
                }
            }
        }
    }
}
