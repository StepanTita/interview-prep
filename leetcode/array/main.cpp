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
const int MOD = 1e9 + 7;

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

// 1814. Count Nice Pairs in an Array

int rev(int n) {
    int res = 0;
    while (n > 0) {
        res = 10 * res + (n % 10);
        n /= 10;
    }

    return res;
}

int countNicePairs(std::vector<int>& nums) {
    // nums[i] - rev(nums[i]) == nums[j] - rev(nums[j])
    int n = nums.size();

    std::unordered_map<int, int> next;
    for (int j = 0; j < n; ++j) {
        ++next[nums[j] - rev(nums[j])];
    }

    int res = 0;
    for (int i = 0; i < n; ++i) {
        --next[nums[i] - rev(nums[i])];
        res = (res + next[nums[i] - rev(nums[i])]) % MOD;
    }

    return res;
}

// 1013. Partition Array Into Three Parts With Equal Sum

bool canThreePartsEqualSum(std::vector<int>& arr) {
    int n = arr.size();

    std::vector<int> prefix(n, 0);
    prefix[0] = arr[0];
    for (int i = 1; i < n; ++i) {
        prefix[i] = prefix[i - 1] + arr[i];
    }

    if (prefix[n - 1] % 3 != 0) return false;

    int part = prefix[n - 1] / 3;

    int i = 0;
    for (; i < n; ++i) {
        if (prefix[i] == part) break;
    }

    if (i == n) return false;

    int j = i + 1;
    for (; j < n; ++j) {
        if (prefix[j] - prefix[i] == part) break;
    }

    if (j == n) return false;

    return n - 1 != j && prefix[n - 1] - prefix[j] == part;
}

// 1018. Binary Prefix Divisible By 5

std::vector<bool> prefixesDivBy5(std::vector<int>& nums) {
    int n = nums.size();

    std::vector<bool> ans(n, 0);

    int val = nums[0];
    ans[0] = !(val % 5);
    for (int i = 1; i < n; ++i) {
        nums[i] = (nums[i - 1] * 2 + nums[i]) % 10;
        ans[i] = !(nums[i] % 5);
    }

    return ans;
}

int main() {
    auto v = std::vector<int>{1, -1, 1, -1};
    canThreePartsEqualSum(v);
    return 0;
}