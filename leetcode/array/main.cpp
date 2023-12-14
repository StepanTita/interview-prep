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

std::vector<int> sortArrayByParityII(std::vector<int>& nums) {
    for (int searcher = 1, waiter = 0; searcher < nums.size(); ++searcher) {
        while (waiter < nums.size() && nums[waiter] % 2 == waiter % 2) {
            ++waiter;
            searcher = std::max(searcher, waiter + 1);
        }
        if (searcher < nums.size() && nums[searcher] % 2 != searcher % 2) {
            std::swap(nums[waiter], nums[searcher]);
        }
    }

    return nums;
}

// 1184. Distance Between Bus Stops

int distanceBetweenBusStops(std::vector<int>& distance, int start, int destination) {
    int n = distance.size();

    int forwardPath = 0;
    for (int i = start; i != destination; i = (i + 1) % n) {
        forwardPath += distance[i];
    }

    int backwardPath = 0;
    for (int i = destination; i != start; i = (i + 1) % n) {
        backwardPath += distance[i];
    }

    return std::min(forwardPath, backwardPath);
}