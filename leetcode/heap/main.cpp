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

//std::vector<std::vector<int>> kSmallestPairs(std::vector<int> &nums1, std::vector<int> &nums2, int k) {
//    int n = nums1.size();
//    int m = nums2.size();
//
//    std::priority_queue<std::vector<int>> q;
//
//    q.push(std::vector<int>{-(nums1[0] + nums2[0]), 0, 0});
//
//    std::set<std::pair<int, int>> visited;
//
//    std::vector<std::vector<int>> res;
//    while (!q.empty() && k--) {
//        auto curr = q.top();
//
//        int i = curr[1];
//        int j = curr[2];
//
//        q.pop();
//        res.emplace_back(std::vector<int>{nums1[i], nums2[j]});
//
//        if (i + 1 < n && !visited.contains({i + 1, j})) {
//            q.push(std::vector<int>{-(nums1[i + 1] + nums2[j]), i + 1, j});
//            visited.insert({i + 1, j});
//        }
//
//        if (j + 1 < m && !visited.contains({i, j + 1})) {
//            q.push(std::vector<int>{-(nums1[i] + nums2[j + 1]), i, j + 1});
//            visited.insert({i, j + 1});
//        }
//    }
//
//    return res;
//}


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

// 1439. Find the Kth Smallest Sum of a Matrix With Sorted Rows

int kthSmallest(std::vector<std::vector<int>> &mat, int k) {
    int n = mat.size();
    int m = mat[0].size();

    std::set<std::vector<int>> visited;
    std::priority_queue<std::pair<int, std::vector<int>>> pq;

    int sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += mat[i][0];
    }

    pq.push({-sum, std::vector<int>(n, 0)});

    std::vector<int> res;

    while (res.size() < k) {
        auto [currSum, idxs] = pq.top();
        res.emplace_back(-pq.top().first);

        pq.pop();

        for (int i = 0; i < n; ++i) {
            if (idxs[i] + 1 >= m) continue;

            std::vector<int> next(idxs.begin(), idxs.end());
            next[i] = idxs[i] + 1;

            if (visited.find(next) != visited.end()) continue;

            pq.push({currSum + mat[i][idxs[i]] - mat[i][next[i]], next});
            visited.insert(next);
        }
    }

    return res.back();
}

// 239. Sliding Window Maximum

// O(n log n)
std::vector<int> maxSlidingWindow(std::vector<int>& nums, int k) {
    int n = nums.size();

    std::map<int, int> valCount;

    for (int i = 0; i < k; ++i) {
        ++valCount[nums[i]];
    }

    std::vector<int> res;
    res.emplace_back(valCount.rbegin()->first);

    int l = 0;
    for (int r = k; r < n; ++r, ++l) {
        if (--valCount[nums[l]] == 0) {
            valCount.erase(nums[l]);
        }

        ++valCount[nums[r]];

        res.emplace_back(valCount.rbegin()->first);
    }

    return res;
}

// O(n)
std::vector<int> maxSlidingWindow2(std::vector<int>& nums, int k) {
    int n = nums.size();

    std::vector<int> res;
    std::deque<int> window;
    for (int i = 0; i < n; ++i) {
        if (!window.empty() && window.front() < i - k + 1) {
            window.pop_front();
        }

        while (!window.empty() && nums[window.back()] < nums[i]) {
            window.pop_back();
        }

        window.push_back(i);

        if (i >= k - 1)
            res.emplace_back(nums[window.front()]);
    }

    return res;
}

// 692. Top K Frequent Words

std::vector<std::string> topKFrequent(std::vector<std::string>& words, int k) {
    // hashmap[word] = count ~ O(1)

    std::unordered_map<std::string, int> freq;
    for (auto w : words) {
        ++freq[w];
    }

    auto compare = [](std::pair<int, std::string> a, std::pair<int, std::string> b) {
        return a.first < b.first || (a.first == b.first && b.second < a.second);
    };

    std::priority_queue<std::pair<int, std::string>, std::vector<std::pair<int, std::string>>, decltype(compare)> pq(compare);

    for (auto [k, v] : freq) {
        pq.push(std::make_pair(v, k));
    }

    std::vector<std::string> res;
    for (int i = 0; i < k; ++i) {
        res.emplace_back(pq.top().second);
        pq.pop();
    }

    return res;
}

// 630. Course Schedule III

int scheduleCourse(std::vector<std::vector<int>>& courses) {
    std::sort(courses.begin(), courses.end(), [](const std::vector<int> &a, const std::vector<int> &b) {
        return a[1] < b[1];
    });

    int n = courses.size();

    std::priority_queue<int> takenCourses;

    int totalTime = 0;
    for (int i = 0; i < n; ++i) {
        int currTime = courses[i][0];
        int deadline = courses[i][1];

        takenCourses.push(currTime);

        totalTime += currTime;

        if (totalTime > deadline) {
            totalTime -= takenCourses.top();
            takenCourses.pop();
        }
    }

    return takenCourses.size();
}

int main() {
    return 0;
}