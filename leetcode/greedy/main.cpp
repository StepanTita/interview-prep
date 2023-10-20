#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <climits>
#include <string>
#include <cctype>
#include <stack>
#include <chrono>
#include <queue>

const int INF = 1e9;

// 2900. Longest Unequal Adjacent Groups Subsequence I

std::vector<std::string> getWordsInLongestSubsequence(int n, std::vector<std::string> &words, std::vector<int> &groups) {
    // so I need to select the longest subsequence from words, such that
    // in groups the next element of this subsequence is different
    std::vector<std::string> odd;

    int prev = groups[0];

    odd.emplace_back(words[0]);
    for (int i = 1; i < n; ++i) {
        if (prev != groups[i]) {
            odd.emplace_back(words[i]);
            prev = groups[i];
        }
    }

    return odd;
}