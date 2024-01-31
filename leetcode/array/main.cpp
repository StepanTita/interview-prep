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

// 1732. Find the Highest Altitude

int largestAltitude(std::vector<int>& gain) {
    int maxSum = 0;
    int currSum = 0;

    for (int i = 0; i < gain.size(); ++i) {
        currSum += gain[i];

        maxSum = std::max(maxSum, currSum);
    }

    return maxSum;
}

// 961. N-Repeated Element in Size 2N Array

int repeatedNTimes(std::vector<int>& nums) {
    for (int i = 1; i < nums.size(); ++i) {
        if (nums[i - 1] == nums[i]) {
            return nums[i];
        } else if (i - 2 >= 0 && nums[i - 2] == nums[i]) {
            return nums[i];
        }
    }
    return nums[0];
}

// 1365. How Many Numbers Are Smaller Than the Current Number

std::vector<int> smallerNumbersThanCurrent(std::vector<int>& nums) {
    int n = nums.size();

    std::vector<std::pair<int, int>> places(n);
    for (int i = 0; i < n; ++i) {
        places[i] = std::pair<int, int>{nums[i], i};
    }

    std::sort(places.begin(), places.end());

    std::vector<int> res(n, 0);

    int dups = 0;
    for (int i = 0; i < n; ++i) {
        if (i - 1 >= 0 && places[i - 1].first == places[i].first) {
            ++dups;
        } else {
            dups = 0;
        }
        res[places[i].second] = i - dups;
    }

    return res;
}