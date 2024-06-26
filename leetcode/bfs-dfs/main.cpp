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

// 838. Push Dominoes

std::string pushDominoes(std::string dominoes) {
    std::queue<int> q;

    int n = dominoes.length();

    for (int i = 0; i < n; ++i) {
        if (dominoes[i] != '.') {
            q.push(i);
        }
    }

    while (!q.empty()) {
        auto curr = q.front();
        q.pop();

        if (dominoes[curr] == 'L') {
            if (curr - 1 >= 0 && dominoes[curr - 1] == '.') {
                dominoes[curr - 1] = 'L';
                q.push(curr - 1);
            }
        } else {
            if (curr + 1 < n && dominoes[curr + 1] == '.') {
                if (curr + 2 >= n || dominoes[curr + 2] != 'L') {
                    dominoes[curr + 1] = 'R';
                    q.push(curr + 1);
                } else if (curr + 2 < n) {
                    q.pop();
                }
            }
        }
    }

    return dominoes;
}

// 854. K-Similar Strings

int kSimilarity(std::string s1, std::string s2) {
    int n = s1.length();

    std::queue<std::string> q;
    q.push(s1);

    std::unordered_map<std::string, int> dist;
    dist[s1] = 0;

    while (!q.empty()) {
        auto curr = q.front();
        q.pop();

        int i = 0;
        while (i < n && curr[i] == s2[i]) ++i;

        for (int j = i + 1; j < n; ++j) {
            if (curr[j] == s2[i] && s2[j] != curr[j]) {
                std::string next = curr;
                std::swap(next[i], next[j]);

                if (dist.contains(next)) continue;

                q.push(next);

                dist[next] = dist[curr] + 1;

                if (next == s2) return dist[next];
            }
        }
    }

    return dist[s2];
}

// 1028. Recover a Tree From Preorder Traversal

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode() : val(0), left(nullptr), right(nullptr) {}

    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}

    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

TreeNode *recoverFromPreorder(std::string traversal) {
    int depth = 0;
    std::string val = "";

    std::unordered_map<int, TreeNode *> prev;
    for (int i = 0; i < traversal.length();) {
        if (traversal[i] == '-') {
            ++depth;
            ++i;
        } else {
            val = "";
            while (i < traversal.length() && traversal[i] != '-') {
                val += traversal[i];
                ++i;
            }

            int num = std::stoi(val);
            prev[depth] = new TreeNode(num);

            if (depth != 0) {
                if (prev[depth - 1]->left) {
                    prev[depth - 1]->right = prev[depth];
                } else {
                    prev[depth - 1]->left = prev[depth];
                }
            }

            depth = 0;
        }
    }

    return prev[0];
}

// 1162. As Far from Land as Possible

bool isValid(int n, int m, int i, int j) {
    if (i < 0 || j < 0 || i >= n || j >= m) {
        return false;
    }
    return true;
}

int maxDistance(std::vector<std::vector<int>> &grid) {
    int n = grid.size();
    int m = grid[0].size();

    std::queue<std::pair<int, int>> q;
    std::vector<std::vector<int>> dist(n, std::vector<int>(m, -1));

    int isls = 0;
    int water = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (grid[i][j] == 1) {
                dist[i][j] = 0;
                q.push(std::pair<int, int>{i, j});
                ++isls;
            } else {
                ++water;
            }
        }
    }

    if (isls == 0 || water == 0) return -1;

    while (!q.empty()) {
        auto [i, j] = q.front();
        q.pop();

        std::vector<std::pair<int, int>> dirs{
                {-1, 0},
                {0,  -1},
                {0,  1},
                {1,  0}
        };
        for (auto [di, dj]: dirs) {
            if (isValid(n, m, i + di, j + dj) && grid[i + di][j + dj] == 0 && dist[i + di][j + dj] == -1) {
                dist[i + di][j + dj] = dist[i][j] + std::abs(di) + std::abs(dj);
                q.push({i + di, j + dj});
            }
        }
    }

    int maxDist = -1;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            maxDist = std::max(maxDist, dist[i][j]);
        }
    }

    return maxDist;
}

// 103. Binary Tree Zigzag Level Order Traversal

void reverseQueue(std::queue<TreeNode *> &q) {
    std::stack<TreeNode *> st;
    while (!q.empty()) {
        st.push(q.front());
        q.pop();
    }

    while (!st.empty()) {
        q.push(st.top());
        st.pop();
    }
}

std::vector<std::vector<int>> zigzagLevelOrder(TreeNode *root) {
    std::queue<TreeNode *> q;

    std::vector<std::vector<int>> res;
    if (root == NULL) return res;

    res.emplace_back(std::vector<int>());

    bool forward = true;

    q.push(root);
    q.push(NULL);
    while (!q.empty()) {
        auto curr = q.front();
        q.pop();

        res.back().emplace_back(curr->val);

        if (forward) {
            if (curr->left != NULL) {
                q.push(curr->left);
            }
            if (curr->right != NULL) {
                q.push(curr->right);
            }
        } else {
            if (curr->right != NULL) {
                q.push(curr->right);
            }
            if (curr->left != NULL) {
                q.push(curr->left);
            }
        }

        if (q.front() == NULL) {
            q.pop();

            reverseQueue(q);
            forward = !forward;

            q.push(NULL);
            if (q.front() == NULL) {
                break;
            }

            res.emplace_back(std::vector<int>());
        }
    }

    return res;
}

// 463. Island Perimeter

enum Mark {
    Empty = 0,
    Island = 1,
    Marked = 2
};

bool empty(std::vector<std::vector<int>> &grid, int i, int j) {
    if (i < 0 || i >= grid.size()) {
        return true;
    }

    if (j < 0 || j >= grid[i].size()) {
        return true;
    }

    return grid[i][j] == Empty;
};

int islandPerimeter(std::vector<std::vector<int>> &grid) {
    int res = 0;
    for (int i = 0; i < grid.size(); ++i) {
        for (int j = 0; j < grid[i].size(); ++j) {
            if (grid[i][j]) {
                if (empty(grid, i - 1, j)) {
                    ++res;
                }
                if (empty(grid, i + 1, j)) {
                    ++res;
                }
                if (empty(grid, i, j - 1)) {
                    ++res;
                }
                if (empty(grid, i, j + 1)) {
                    ++res;
                }
                grid[i][j] = Marked;
            }
        }
    }
    return res;
}

// 1305. All Elements in Two Binary Search Trees

std::vector<int> getAllElements(TreeNode* root1, TreeNode* root2) {
    std::stack<TreeNode*> s1, s2;
    std::vector<int> res;

    while (root1 || root2 || !s1.empty() || !s2.empty()) {
        while (root1 != NULL) {
            s1.push(root1);
            root1 = root1->left;
        }

        while (root2 != NULL) {
            s2.push(root2);
            root2 = root2->left;
        }

        if (s2.empty() || (!s1.empty() && s1.top()->val <= s2.top()->val)) {
            root1 = s1.top();
            s1.pop();

            res.emplace_back(root1->val);
            root1 = root1->right;
        } else {
            root2 = s2.top();
            s2.pop();

            res.emplace_back(root2->val);
            root2 = root2->right;
        }
    }

    return res;
}

// 965. Univalued Binary Tree

bool isUnivalTree(TreeNode* root) {
    if (root == NULL) return true;

    if (root->left != NULL && root->left->val != root->val) return false;
    if (root->right != NULL && root->right->val != root->val) return false;

    return isUnivalTree(root->left) && isUnivalTree(root->right);
}

int main() {
    return 0;
}