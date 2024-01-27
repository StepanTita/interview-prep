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

// 200. Number of Islands

class NumberOfIslands {
private:
    std::vector<int> parent;
    std::vector<int> rank;
public:
    int find(int x) {
        if (parent[x] == -1) return parent[x] = x;
        if (parent[x] == x) return x;
        return parent[x] = find(parent[x]);
    }

    void unite(int a, int b) {
        int pa = find(a);
        int pb = find(b);

        if (pa == pb) return;

        if (rank[pa] > rank[pb]) {
            parent[pb] = pa;
        } else if (rank[pa] < rank[pb]) {
            parent[pa] = pb;
        } else {
            parent[pa] = pb;
            ++rank[pb];
        }
    }

    int toIndex(int i, int j) {
        return i | (j << 9);
    }

    int numIslands(std::vector<std::vector<char>>& grid) {
        int n = grid.size();
        int m = grid[0].size();

        parent = std::vector<int>(1 << 18, -1);
        rank = std::vector<int>(1 << 18, 0);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (grid[i][j] != '1') continue;

                if (i - 1 >= 0 && grid[i - 1][j] == '1') {
                    unite(toIndex(i - 1, j), toIndex(i, j));
                }
                if (i + 1 < n && grid[i + 1][j] == '1') {
                    unite(toIndex(i, j), toIndex(i + 1, j));
                }

                if (j - 1 >= 0 && grid[i][j - 1] == '1') {
                    unite(toIndex(i, j), toIndex(i, j - 1));
                }
                if (j + 1 < m && grid[i][j + 1] == '1') {
                    unite(toIndex(i, j), toIndex(i, j + 1));
                }
            }
        }

        std::unordered_set<int> islands;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (grid[i][j] != '1') continue;

                islands.insert(find(toIndex(i, j)));
            }
        }

        return islands.size();
    }
};