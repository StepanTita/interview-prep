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

class RandomizedSet {
private:
    std::vector<int> container;
    std::unordered_map<int, int> lookup;
    int size;
public:
    RandomizedSet() {
        container = std::vector<int>(2 * 1e5 + 1, 0);
        size = 0;
    }

    bool insert(int val) {
        if (lookup.find(val) != lookup.end()) return false;

        lookup[val] = size;
        container[size] = val;
        size++;
        return true;
    }

    bool remove(int val) {
        if (lookup.find(val) == lookup.end()) return false;

        int idx = lookup[val];
        lookup.erase(val);

        if (idx < size - 1) {
            container[idx] = container[size - 1];
            lookup[container[size - 1]] = idx;
        }

        size--;
        return true;
    }

    int getRandom() {
        int num = std::rand() % size;
        return container[num];
    }
};

// 855. Exam Room

class ExamRoom {
private:
    int n;
    std::set<int> room;
public:
    // 1 0 0 0 1 0 0 0 1
    ExamRoom(int n) {
        this->n = n;
    }

    int seat() {
        int dist = 0;
        int pos = 0;

        if (!room.empty()) {
            auto prev = room.begin();
            dist = *prev;

            auto curr = next(prev);

            while (curr != room.end()) {
                int mid = (*curr - *prev) / 2;

                if (dist < mid) {
                    dist = mid;

                    pos = *prev + dist;
                }

                prev = curr;
                curr = next(curr);
            }

            if (dist < ((n - 1) - *(room.rbegin()))) {
                pos = n - 1;
            }
        }

        room.insert(pos);
        return pos;
    }

    void leave(int p) {
        // when leave compare freed segment with the current one
        room.erase(p);
    }
};

// 1396. Design Underground System

class UndergroundSystem {
private:
    std::unordered_map<std::string, std::unordered_map<std::string, int>> time;
    std::unordered_map<std::string, std::unordered_map<std::string, int>> count;

    std::unordered_map<int, std::pair<std::string, int>> checkIns;
public:
    UndergroundSystem() {
    }

    void checkIn(int id, std::string stationName, int t) {
        checkIns[id] = std::make_pair(stationName, t);
    }

    void checkOut(int id, std::string endStation, int t) {
        auto [startStation, startTime] = checkIns[id];

        time[startStation][endStation] += t - startTime;
        ++count[startStation][endStation];

        checkIns.erase(id);
    }

    double getAverageTime(std::string startStation, std::string endStation) {
        return double(time[startStation][endStation]) / double(count[startStation][endStation]);
    }
};

// 528. Random Pick with Weight

class Solution {
private:
    int n = 0;
    int sum = 0;
    std::vector<int> prefixSum;
public:
    Solution(std::vector<int> &w) {
        n = w.size();
        prefixSum = std::vector<int>(n, 0);

        prefixSum[0] = w[0];
        for (int i = 1; i < w.size(); ++i) {
            prefixSum[i] = prefixSum[i - 1] + w[i];
        }
    }

    int pickIndex() {
        double v = ((double) rand() / (RAND_MAX)) * prefixSum[n - 1];
        return std::lower_bound(prefixSum.begin(), prefixSum.end(), v) - prefixSum.begin();
    }
};

int main() {
    ExamRoom *examRoom = new ExamRoom(10);
    examRoom->seat(); // return 0, no one is in the room, then the student sits at seat number 0.
    examRoom->seat(); // return 9, the student sits at the last seat number 9.
    examRoom->seat(); // return 4, the student sits at the last seat number 4.
    examRoom->seat(); // return 2, the student sits at the last seat number 2.
    examRoom->leave(4);
    examRoom->seat(); // return 5, the student sits at the last seat number 5.
    return 0;
}