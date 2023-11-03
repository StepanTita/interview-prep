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
