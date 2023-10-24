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

// 1131. Maximum of Absolute Value Expression

int findMax(std::vector<int>& arr1, std::vector<int>& arr2, int s1, int s2) {
    int n = arr1.size();
    int maxVal = -INF;
    int minVal = INF;

    for (int i = 0; i < n; ++i) {
        maxVal = std::max(maxVal, arr1[i] + s1 * arr2[i] + s2 * i);
        minVal = std::min(minVal, arr1[i] + s1 * arr2[i] + s2 * i);
    }

    return maxVal - minVal;
}

int maxAbsValExpr(std::vector<int>& arr1, std::vector<int>& arr2) {
    auto signs = std::vector<std::pair<int, int>>{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

    int ans = 0;
    for (auto [s1, s2] : signs) {
        ans = std::max(ans, findMax(arr1, arr2, s1, s2));
    }

    return ans;
}