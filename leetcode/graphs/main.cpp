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

        std::unordered_set<std::string> visited;

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

int findCheapestPrice(int n, std::vector<std::vector<int>> &flights, int src, int dst, int K) {
    std::vector<std::vector<std::pair<int, int>>> adjList(n);

    for (auto flight: flights) {
        int from = flight[0];
        int to = flight[1];
        int cost = flight[2];
        adjList[from].emplace_back(std::pair<int, int>{to, cost});
    }

    std::vector<std::vector<int>> dp(n, std::vector<int>(K + 2, INF));
    dp[src][0] = 0;

    for (int k = 0; k <= K; ++k) {
        for (int i = 0; i < n; ++i) {
            for (auto [next, cost]: adjList[i]) {
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

// 1145. Binary Tree Coloring Game

int subtreeSize(TreeNode *curr, int n, int x, bool &ans) {
    if (curr == NULL) return 0;

    int left = subtreeSize(curr->left, n, x, ans);
    int right = subtreeSize(curr->right, n, x, ans);

    if (curr->val == x) {
        int prev = n - left - right - 1;
        // parent
        if (prev > left + right + 1) {
            ans = true;
        } else if (left > (n - left)) { // left
            ans = true;
        } else if (right > (n - right)) { // right
            ans = true;
        }
    }

    return left + right + 1;
}

bool btreeGameWinningMove(TreeNode *root, int n, int x) {
    bool ans = false;
    subtreeSize(root, n, x, ans);
    return ans;
}

// 1123. Lowest Common Ancestor of Deepest Leaves

std::pair<TreeNode *, int> findLCA(TreeNode *curr) {
    if (curr == NULL) return std::make_pair<TreeNode *, int>(NULL, 0);

    auto [left, ldepth] = findLCA(curr->left);
    auto [right, rdepth] = findLCA(curr->right);

    if (ldepth < rdepth) {
        return std::make_pair(right, rdepth + 1);
    } else if (ldepth > rdepth) {
        return std::make_pair(left, ldepth + 1);
    }

    return std::make_pair(curr, ldepth + 1);
}

TreeNode *lcaDeepestLeaves(TreeNode *root) {
    return findLCA(root).first;
}

// 1026. Maximum Difference Between Node and Ancestor

int dfs(TreeNode *curr, int currMin, int currMax) {
    if (curr == NULL) return 0;

    int left = dfs(curr->left, std::min(curr->val, currMin), std::max(curr->val, currMax));
    int right = dfs(curr->right, std::min(curr->val, currMin), std::max(curr->val, currMax));

    return std::max({
                            left,
                            right,
                            std::abs(curr->val - currMin),
                            std::abs(curr->val - currMax)
                    });
}

int maxAncestorDiff(TreeNode *root) {
    return dfs(root, root->val, root->val);
}

// 743. Network Delay Time

int networkDelayTime(std::vector<std::vector<int>> &times, int n, int k) {
    --k;
    std::vector<std::vector<int>> adj_mat(n, std::vector<int>(n, INF));

    for (auto edge: times) {
        int u = edge[0] - 1, v = edge[1] - 1, w = edge[2];
        adj_mat[u][v] = w;
    }

    std::vector<int> dist(n, INF);
    std::vector<bool> visited(n, false);
    dist[k] = 0;

    std::priority_queue<std::pair<int, int>> pq;
    pq.push(std::make_pair(0, k));

    while (!pq.empty()) {
        auto [_, u] = pq.top();
        pq.pop();

        if (visited[u]) continue;
        visited[u] = true;

        for (int v = 0; v < n; ++v) {
            int w = adj_mat[u][v];
            if (w == INF) continue;

            if (dist[v] > dist[u] + w) {
                dist[v] = dist[u] + w;
                pq.push(std::make_pair(-dist[v], v));
            }
        }
    }

    int total = 0;
    for (int i = 0; i < n; ++i) {
        if (dist[i] >= INF) return -1;
        total = std::max(total, dist[i]);
    }

    return total;
}

bool isValid(int i, int j, int n, int m) {
    if (i < 0 || i >= n || j < 0 || j >= m) return false;
    return true;
}

int minimumEffortPath(std::vector<std::vector<int>> &heights) {
    int n = heights.size();
    int m = heights[0].size();

    std::vector<std::vector<int>> dist(n, std::vector<int>(m, INF));
    std::vector<std::vector<bool>> visited(n, std::vector<bool>(m, false));

    // diff, i, j
    std::priority_queue<std::vector<int>> pq;
    pq.push({0, 0, 0});

    dist[0][0] = 0;

    std::vector<std::pair<int, int>> dirs{{-1, 0},
                                          {0,  -1},
                                          {0,  1},
                                          {1,  0}};

    while (!pq.empty()) {
        auto u = pq.top();
        pq.pop();

        int i = u[1];
        int j = u[2];

        if (visited[i][j]) continue;
        visited[i][j] = true;

        for (auto [di, dj]: dirs) {
            if (!isValid(i + di, j + dj, n, m)) continue;

            int w = std::abs(heights[i][j] - heights[i + di][j + dj]);
            if (dist[i + di][j + dj] > std::max(dist[i][j], w)) {
                dist[i + di][j + dj] = std::max(dist[i][j], w);
                pq.push({-dist[i + di][j + dj], i + di, j + dj});
            }
        }
    }

    return dist[n - 1][m - 1];
}

// 133. Clone Graph

class Node {
public:
    int val;
    std::vector<Node *> neighbors;

    Node() {
        val = 0;
        neighbors = std::vector<Node *>();
    }

    Node(int _val) {
        val = _val;
        neighbors = std::vector<Node *>();
    }

    Node(int _val, std::vector<Node *> _neighbors) {
        val = _val;
        neighbors = _neighbors;
    }
};

Node *dfs(Node *node, std::vector<Node *> &visited) {
    auto newNode = new Node(node->val, node->neighbors);

    visited[node->val] = newNode;

    for (int i = 0; i < newNode->neighbors.size(); ++i) {
        if (visited[newNode->neighbors[i]->val] != NULL) {
            newNode->neighbors[i] = visited[newNode->neighbors[i]->val];
            continue;
        }
        newNode->neighbors[i] = dfs(newNode->neighbors[i], visited);
    }

    return newNode;
}

Node *cloneGraph(Node *node) {
    if (node == NULL) return NULL;

    std::vector<Node *> visited(101, NULL);

    return dfs(node, visited);
}

// 117. Populating Next Right Pointers in Each Node II

struct ConnTreeNode {
    int val;
    ConnTreeNode *left;
    ConnTreeNode *right;
    ConnTreeNode *next;
};

ConnTreeNode *connect(ConnTreeNode *root) {
    if (root == NULL) return NULL;

    auto head = root;

    for (; root != NULL; root = root->next) {
        auto dummy = new ConnTreeNode();
        auto curr = dummy;
        for (; root != NULL; root = root->next) {
            if (root->left != NULL) {
                curr->next = root->left;
                curr = curr->next;
            }
            if (root->right != NULL) {
                curr->next = root->right;
                curr = curr->next;
            }
        }
        root = dummy;
    }
    return head;
}

// 236. Lowest Common Ancestor of a Binary Tree

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};

TreeNode* lowestCommonAncestor(TreeNode* curr, TreeNode* p, TreeNode* q) {
    if (curr == NULL || curr == p || curr == q) return curr;

    auto left = lowestCommonAncestor(curr->left, q, p);
    auto right = lowestCommonAncestor(curr->right, q, p);

    if (left != NULL && right != NULL) return curr;
    if (left != NULL) return left;
    return right;
}

// 797. All Paths From Source to Target

void dfs(
        int curr,
        int target,
        std::vector<std::vector<int>> &graph,
        std::vector<int> &path,
        std::vector<std::vector<int>> &res
) {
    path.emplace_back(curr);

    if (curr == target) {
        res.emplace_back(std::vector<int>(path.begin(), path.end()));
        path.pop_back();
        return;
    }

    for (int next : graph[curr]) {
        dfs(next, target, graph, path, res);
    }

    path.pop_back();
}

std::vector<std::vector<int>> allPathsSourceTarget(std::vector<std::vector<int>>& graph) {
    int n = graph.size();

    std::vector<int> path;
    std::vector<std::vector<int>> res;

    dfs(0, n - 1, graph, path, res);

    return res;
}

// 1584. Min Cost to Connect All Points

class MST {
private:
    std::vector<int> parent;
    std::vector<int> rank;
public:
    int find(int x) {
        if (parent[x] == -1) return parent[x] = x;
        else if (parent[x] == x) return x;
        else return parent[x] = find(parent[x]);
    }

    int unite(int a, int b) {
        int pa = find(a);
        int pb = find(b);

        if (pa == pb) return pa;

        int p = pa;

        if (rank[pa] > rank[pb]) {
            p = pb;
        } else if (rank[pa] == rank[pb]) {
            rank[pa]++;
        }

        parent[pa] = p;
        parent[pb] = p;
        return p;
    }

    int minCostConnectPoints(std::vector<std::vector<int>>& points) {
        int n = points.size();

        int E = n * n;

        parent = std::vector<int>(E, -1);
        rank = std::vector<int>(E, 0);

        std::vector<std::vector<int>> edges;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                edges.emplace_back(std::vector<int>{abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1]), i, j});
            }
        }

        std::sort(edges.begin(), edges.end());

        int minCost = 0;

        for (auto edge : edges) {
            int w = edge[0];
            int a = edge[1];
            int b = edge[2];

            if (find(a) == find(b)) continue;

            minCost += w;

            unite(a, b);
        }

        return minCost;
    }
};


int main() {
    auto t = std::vector<std::vector<int>>{
            {1, 2, 1},
            {2, 3, 2},
            {1, 3, 4}
    };
    networkDelayTime(t, 3, 1);
    return 0;
}