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

// 215. Kth Largest Element in an Array

int findKthLargest(std::vector<int> &nums, int k) {
    std::priority_queue<int> heap;
    for (auto num: nums) {
        heap.push(num);
    }

    int i = 0;
    int last_max = 0;
    while (heap.size() > 0 && i < k) {
        last_max = heap.top();
        heap.pop();
        ++i;
    }
    return last_max;
}

// 502. IPO

int findMaximizedCapital(int k, int w, std::vector<int> &profits, std::vector<int> &capital) {
    int n = profits.size();

    std::vector<std::pair<int, int>> projects(n);
    for (int i = 0; i < n; ++i) {
        projects[i] = {capital[i], profits[i]};
    }

    std::sort(projects.begin(), projects.end());

    int i = 0;
    std::priority_queue<int> maximizeCapital;

    while (k--) {
        while (i < n && projects[i].first <= w) {
            maximizeCapital.push(projects[i].second);
            ++i;
        }

        if (maximizeCapital.empty()) break;

        w += maximizeCapital.top();
        maximizeCapital.pop();
    }

    return w;
}

// 373. Find K Pairs with Smallest Sums

std::vector<std::vector<int>> kSmallestPairs(std::vector<int> &nums1, std::vector<int> &nums2, int k) {
    int n = nums1.size();
    int m = nums2.size();

    std::priority_queue<std::vector<int>> q;

    q.push(std::vector<int>{-(nums1[0] + nums2[0]), 0, 0});

    std::set<std::pair<int, int>> visited;

    std::vector<std::vector<int>> res;
    while (!q.empty() && k--) {
        auto curr = q.top();

        int i = curr[1];
        int j = curr[2];

        q.pop();
        res.emplace_back(std::vector<int>{nums1[i], nums2[j]});

        if (i + 1 < n && !visited.contains({i + 1, j})) {
            q.push(std::vector<int>{-(nums1[i + 1] + nums2[j]), i + 1, j});
            visited.insert({i + 1, j});
        }

        if (j + 1 < m && !visited.contains({i, j + 1})) {
            q.push(std::vector<int>{-(nums1[i] + nums2[j + 1]), i, j + 1});
            visited.insert({i, j + 1});
        }
    }

    return res;
}


// 295. Find Median from Data Stream

class MedianFinder {
    std::priority_queue<int> lower; // 1, 2
    std::priority_queue<int> upper; // 3
public:
    MedianFinder() {}

    void transfuse() {
        while (upper.size() > lower.size()) {
            lower.push(-upper.top());
            upper.pop();
        }

        while ((lower.size() - upper.size()) > 1) {
            upper.push(-lower.top());
            lower.pop();
        }
    }

    // 1 1 2 3
    void addNum(int num) {
        if (lower.empty()) {
            lower.push(num);
            return;
        }

        if (num <= lower.top()) {
            lower.push(num);
        } else {
            upper.push(-num);
        }

        transfuse();
    }

    double findMedian() {
        if ((lower.size() + upper.size()) % 2 == 0) {
            return ((double) lower.top() - (double) upper.top()) / 2;
        }
        return lower.top();
    }
};

// 857. Minimum Cost to Hire K Workers

double mincostToHireWorkers(std::vector<int> &quality, std::vector<int> &wage, int k) {
    std::vector<std::pair<int, int>> workers;

    for (int i = 0; i < quality.size(); ++i) {
        workers.emplace_back(std::pair<int, int>{quality[i], wage[i]});
    }

    std::sort(workers.begin(), workers.end(), [](std::pair<int, int> a, std::pair<int, int> b) {
        return ((double) a.second / a.first) < ((double) b.second / b.first);
    });

    std::priority_queue<int> pq;

    double sumq = 0;
    double ans = INF;
    for (auto worker: workers) {
        pq.push(worker.first);

        sumq -= worker.first;
        if (pq.size() > k) {
            sumq += pq.top();
            pq.pop();
        }
        if (pq.size() == k) {
            ans = std::min(ans, sumq * ((double) worker.second / (double) worker.first));
        }
    }

    return ans;
}

int main() {
    auto a = std::vector<int>{10,20,5};
    auto b = std::vector<int>{70,50,30};
    std::cout << mincostToHireWorkers(a, b, 2) << std::endl;
    return 0;
}