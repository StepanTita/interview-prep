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

// 146. LRU Cache

struct LNode {
    int key;
    int val;

    LNode *prev;
    LNode *next;

    LNode(int key, int value) {
        this->key = key;
        this->val = value;

        prev = NULL;
        next = NULL;
    }
};

class LRUCache {
private:
    int cap;
    int len;

    LNode *head;
    LNode *tail;

    std::unordered_map<int, LNode *> lruMap;
public:
    LRUCache(int capacity) {
        cap = capacity;
        len = 0;

        head = NULL;
        tail = NULL;
    }

    LNode *getNode(int key) {
        if (!lruMap.contains(key)) return NULL;

        auto curr = lruMap[key];

        if (head != tail && curr != head) {
            auto prev = curr->prev;
            auto next = curr->next;

            if (next != NULL) {
                next->prev = curr->prev;
            }

            prev->next = next;
            curr->next = head;
            head->prev = curr;
            curr->prev = NULL;

            head = curr;
            if (curr == tail) {
                tail = prev;
            }
        }

        return curr;
    }

    int get(int key) {
        auto node = getNode(key);
        if (node == NULL) return -1;

        return node->val;
    }

    void put(int key, int value) {
        if (lruMap.contains(key)) {
            auto curr = getNode(key);
            curr->val = value;
            return;
        }

        auto curr = new LNode(key, value);

        if (head == NULL) {
            head = curr;
            tail = head;
        } else {
            curr->next = head;
            head->prev = curr;
            head = curr;
        }
        lruMap[key] = curr;
        ++len;

        if (len > cap) {
            --len;
            lruMap.erase(tail->key);

            tail = tail->prev;
            if (tail == NULL) {
                head = NULL;
            } else {
                tail->next = NULL;
            }
        }
    }
};

// 208. Implement Trie (Prefix Tree)

struct Node {
    std::unordered_map<char, Node *> next;
    bool isTerminal;

    Node(bool terminal = false) : isTerminal(terminal) {}
};

class Trie {
private:
    Node *head;
public:
    Trie() {
        head = new Node();
    }

    void insert(std::string word) {
        auto curr = head;
        for (char c: word) {
            if (!curr->next.contains(c)) {
                curr->next[c] = new Node();
            }

            curr = curr->next[c];
        }

        curr->isTerminal = true;
    }

    bool search(std::string word) {
        auto curr = head;

        for (char c: word) {
            if (!curr->next.contains(c)) return false;
            curr = curr->next[c];
        }
        return curr->isTerminal;
    }

    bool startsWith(std::string prefix) {
        auto curr = head;

        for (char c: prefix) {
            if (!curr->next.contains(c)) return false;
            curr = curr->next[c];
        }

        return true;
    }
};

// 911. Online Election

class TopVotedCandidate {
public:
    const int INF = 1e9;

    std::vector<int> polls;
    std::vector<int> times;

    TopVotedCandidate(std::vector<int> &persons, std::vector<int> &times) {
        int n = persons.size();

        this->times = times;

        std::unordered_map<int, int> votes;

        int max_votes = 0;
        int max_person = 0;

        polls = std::vector<int>(n, INF);
        for (int i = 0; i < n; ++i) {
            ++votes[persons[i]];
            if (max_votes <= votes[persons[i]]) {
                max_votes = votes[persons[i]];
                max_person = persons[i];
            }

            polls[i] = max_person;
        }
    }

    int q(int t) {
        int i = std::lower_bound(times.begin(), times.end(), t) - times.begin();
        if (i >= times.size()) return polls.back();
        if (times[i] > t) {
            return polls[i - 1];
        }
        return polls[i];
    }
};

int main() {
    LRUCache *cache = new LRUCache(3);
    cache->put(1, 1);
    cache->put(2, 2);
    cache->put(3, 3);
    cache->put(4, 4);
    cache->get(4);
    cache->get(3);
    cache->get(2);
    cache->get(1);
    cache->put(5, 5);
    cache->get(1);
    cache->get(2);
    cache->get(3);
    cache->get(4);
    cache->get(5);
    return 0;
}