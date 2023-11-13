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