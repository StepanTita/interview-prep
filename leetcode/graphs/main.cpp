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

// 399. Evaluate Division

double find(
        std::string &curr,
        std::string &target,
        std::unordered_map<std::string, std::vector<std::string>> &adj_nodes,
        std::unordered_map<std::string, std::vector<double>> &adj_values,
        std::unordered_set<std::string> &visited
) {
    if (curr == target) return 1.0;

    for (int i = 0; i < adj_nodes[curr].size(); ++i) {
        auto next = adj_nodes[curr][i];

        if (visited.contains(next)) continue;

        visited.insert(next);

        auto rem = find(next, target, adj_nodes, adj_values, visited);
        if (rem != -1.0) {
            return adj_values[curr][i] * rem;
        }
    }

    return -1.0;
}

std::vector<double> calcEquation(
        std::vector<std::vector<std::string>> &equations,
        std::vector<double> &values,
        std::vector<std::vector<std::string>> &queries
) {
    std::unordered_map<std::string, std::vector<std::string>> adj_nodes;
    std::unordered_map<std::string, std::vector<double>> adj_values;

    for (int i = 0; i < equations.size(); ++i) {
        auto eq = equations[i];

        adj_nodes[eq[0]].emplace_back(eq[1]);
        adj_nodes[eq[1]].emplace_back(eq[0]);

        adj_values[eq[0]].emplace_back(values[i]);
        adj_values[eq[1]].emplace_back(1.0 / values[i]);
    }

    std::vector<double> res;
    for (auto &q: queries) {
        auto source = q[0];
        auto target = q[1];

        if (!adj_nodes.contains(source) || !adj_nodes.contains(target)) {
            res.emplace_back(-1.0);
            continue;
        }

        std::unordered_set < std::string > visited;

        res.emplace_back(find(source, target, adj_nodes, adj_values, visited));
    }

    return res;
}

// 1305. All Elements in Two Binary Search Trees

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode() : val(0), left(nullptr), right(nullptr) {}

    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}

    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

void postOrder(TreeNode *curr, std::stack<int> &out) {
    if (curr == NULL) {
        return;
    }

    postOrder(curr->right, out);
    out.push(curr->val);
    postOrder(curr->left, out);
}

std::vector<int> getAllElements(TreeNode *a, TreeNode *b) {
    std::vector<int> res;

    std::stack<int> postA;
    std::stack<int> postB;

    postOrder(a, postA);
    postOrder(b, postB);

    while (!postA.empty() && !postB.empty()) {
        while (!postA.empty() && postA.top() <= postB.top()) {
            res.emplace_back(postA.top());
            postA.pop();
        }

        if (postA.empty()) break;

        while (!postB.empty() && postB.top() < postA.top()) {
            res.emplace_back(postB.top());
            postB.pop();
        }
    }

    while (!postA.empty()) {
        res.emplace_back(postA.top());
        postA.pop();
    }

    while (!postB.empty()) {
        res.emplace_back(postB.top());
        postB.pop();
    }

    return res;
}

// 207. Course Schedule

bool dfs(
        std::vector<std::vector<int>> &adj,
        int curr,
        std::vector<bool> &visited,
        std::vector<bool> &inStack
) {
    visited[curr] = true;
    inStack[curr] = true;

    for (int next: adj[curr]) {
        if (!visited[next] && dfs(adj, next, visited, inStack)) {
            return true;
        } else if (inStack[next]) return true;
    }

    inStack[curr] = false;

    return false;
}

bool canFinish(int n, std::vector<std::vector<int>> &prerequisites) {
    std::vector<std::vector<int>> adj(n, std::vector<int>());
    for (auto p: prerequisites) {
        adj[p[1]].emplace_back(p[0]);
    }

    std::vector<bool> visited(n);
    std::vector<bool> inStack(n);
    for (int i = 0; i < n; i++) {
        if (dfs(adj, i, visited, inStack)) {
            return false;
        }
    }

    return true;
}

// 210. Course Schedule II

bool dfs(int curr,
         std::vector<std::vector<int>> &adj,
         std::vector<int> &res,
         std::vector<bool> &visited,
         std::vector<bool> &inStack
) {
    visited[curr] = true;
    inStack[curr] = true;

    for (auto next: adj[curr]) {
        if (!visited[next] && dfs(next, adj, res, visited, inStack)) {
            return true;
        } else if (inStack[next]) {
            return true;
        }
    }

    inStack[curr] = false;
    res.emplace_back(curr);

    return false;
}

std::vector<int> findOrder(int n, std::vector<std::vector<int>> &prerequisites) {
    std::vector<std::vector<int>> adj(n, std::vector<int>());

    for (auto p: prerequisites) {
        adj[p[1]].emplace_back(p[0]);
    }

    std::vector<int> res;
    std::vector<bool> visited(n, false);
    std::vector<bool> inStack(n, false);

    for (int i = 0; i < n; ++i) {
        if (visited[i]) continue;
        if (dfs(i, adj, res, visited, inStack)) {
            res.clear();
            return res;
        }
    }

    std::reverse(res.begin(), res.end());
    return res;
}

// 124. Binary Tree Maximum Path Sum

int buildMax(TreeNode *curr, int &ans) {
    if (curr == NULL) return -INF;

    // take left + curr
    // take right + curr
    // take left + curr + right
    // take only curr

    auto left = buildMax(curr->left, ans);
    auto right = buildMax(curr->right, ans);

    curr->val = std::max(
            {
                    left + curr->val,
                    right + curr->val,
                    curr->val
            }
    );

    ans = std::max({ans, curr->val, left + curr->val + right});

    return curr->val;
}

int maxPathSum(TreeNode *root) {
    int ans = -INF;
    buildMax(root, ans);
    return ans;
}

// 787. Cheapest Flights Within K Stops

int findCheapestPrice(int n, std::vector<std::vector<int>>& flights, int src, int dst, int K) {
    std::vector<std::vector<std::pair<int, int>>> adjList(n);

    for (auto flight : flights) {
        int from = flight[0];
        int to = flight[1];
        int cost = flight[2];
        adjList[from].emplace_back(std::pair<int, int>{to, cost});
    }

    std::vector<std::vector<int>> dp(n, std::vector<int>(K + 2, INF));
    dp[src][0] = 0;

    for (int k = 0; k <= K; ++k) {
        for (int i = 0; i < n; ++i) {
            for (auto [next, cost] : adjList[i]) {
                dp[next][k + 1] = std::min(dp[next][k + 1], dp[i][k] + cost);
            }
        }
    }

    int ans = INF;

    for (int k = 0; k <= K + 1; ++k) {
        ans = std::min(ans, dp[dst][k]);
    }

    if (ans == INF) return -1;

    return ans;
}


int main() {
    auto t = std::vector<std::vector<int>>{
            {0,1,100},
            {1,2,100},
            {2,0,100},
            {1,3,600},
            {2,3,200}
    };
    std::cout << findCheapestPrice(4, t, 0, 3, 1) << std::endl;
    return 0;
}