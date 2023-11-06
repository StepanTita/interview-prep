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

// 2270. Number of Ways to Split Array

int waysToSplitArray(std::vector<int> &nums) {
    long long totalSum = 0;
    for (int v: nums) {
        totalSum += v;
    }

    int n = nums.size();
    long long rightSum = nums[n - 1];

    int count = 0;
    for (int i = n - 2; i >= 0; --i) {
        if (rightSum <= totalSum - rightSum) {
            ++count;
        }
        rightSum += (long long) nums[i];
    }

    return count;
}