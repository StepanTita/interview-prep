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

// 947. Most Stones Removed with Same Row or Column

class Solution {
private:
    std::unordered_map<int, int> parent;
    std::unordered_map<int, int> rank;
public:

    int find(int x) {
        if (!parent.contains(x)) {
            return parent[x] = x;
        }

        if (parent[x] == x) {
            return x;
        }

        return parent[x] = find(parent[x]);
    }

    int unite(int a, int b) {
        int pa = find(a);
        int pb = find(b);

        if (pa == pb) {
            return pa;
        }

        int p = -1;
        if (rank[pa] > rank[pb]) {
            p = pa;
        } else if (rank[pa] < rank[pb]) {
            p = pb;
        } else {
            p = pa;
            ++rank[pa];
        }

        parent[pa] = p;
        parent[pb] = p;

        return p;
    }

    int removeStones(std::vector<std::vector<int>>& stones) {
        int n = stones.size();

        for (int i = 0; i < n; ++i) {
            unite(stones[i][0], ~stones[i][1]);
        }

        std::unordered_set<int> components;
        for (int i = 0; i < n; ++i) {
            components.insert(find(stones[i][0]));
            components.insert(find(~stones[i][1]));
        }

        return n - components.size();
    }
};