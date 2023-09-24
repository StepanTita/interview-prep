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

// 70. Climbing Stairs
int climb(int n, std::vector<int> &dp) {
    dp[0] = 1;
    dp[1] = 1;
    for (int i = 2; i <= n; ++i) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }

    return dp[n];
}

int climbStairs(int n) {
    std::vector<int> dp(n + 1, 0);
    return climb(n, dp);
}

// 198. House Robber
int rob(std::vector<int> &nums) {
    int n = nums.size();

    if (n == 1) {
        return nums[0];
    }

    std::vector<int> dp(n, 0);

    dp[0] = nums[0];
    dp[1] = std::max(dp[0], nums[1]);

    for (int i = 2; i < n; ++i) {
        dp[i] = std::max(dp[i - 1], dp[i - 2] + nums[i]);
    }

    return dp[n - 1];
}

// 139. Word Break
struct TrieNode {
    std::unordered_map<char, TrieNode *> next;
    bool is_terminal;

    TrieNode() : is_terminal(false) {
        next = std::unordered_map < char, TrieNode * > ();
    }

    bool has(char c) {
        return next.find(c) != next.end();
    }
};

void buildTrie(
        const std::string &word,
        TrieNode *curr
) {
    for (char c: word) {
        if (!curr->has(c)) {
            curr->next[c] = new TrieNode();
        }
        curr = curr->next[c];
    }
    curr->is_terminal = true;
}

const int FAIL = 0;
const int PASS = 1;

int dfs(std::string &s, int i, TrieNode *curr, TrieNode *root, std::vector<int> &dp) {
    if (curr == root && dp[i] != -1) return dp[i];

    dp[i] = 0;

    for (int j = i; j < s.length(); ++j) {
        if (curr->is_terminal) {
            dp[j] = dfs(s, j, root, root, dp);
            dp[i] += dp[j];
        }

        if (!curr->has(s[j])) {
            if (dp[i] > 0) return PASS;
            return FAIL;
        }

        curr = curr->next[s[j]];
    }

    return (dp[i] > 0 || curr->is_terminal) ? PASS : FAIL;
}

bool wordBreak(std::string s, std::vector<std::string> &wordDict) {
    TrieNode *root = new TrieNode();

    for (auto &w: wordDict) {
        buildTrie(w, root);
    }

    std::vector<int> dp(s.length(), -1);

    return dfs(s, 0, root, root, dp);
}

// 894. All Possible Full Binary Trees

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;

    TreeNode() : val(0), left(nullptr), right(nullptr) {}

    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}

    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

std::vector<TreeNode *> allPossibleFBT(int n) {
    if (n % 2 == 0) return std::vector<TreeNode *>{};

    std::vector<std::vector<TreeNode *>> dp(n + 1, std::vector<TreeNode *>());
    dp[1] = std::vector<TreeNode *>{new TreeNode()};

    for (int count = 1; count <= n; count += 2) {
        for (int i = 1; i < count; ++i) {
            int j = count - i - 1;
            for (auto left: dp[i]) {
                for (auto right: dp[j]) {
                    auto root = new TreeNode(0, left, right);
                    dp[count].emplace_back(root);
                }
            }
        }
    }

    return dp[n];
}

// 1646. Get Maximum in Generated Array
int getMaximumGenerated(int n) {
    if (n < 1) return 0;

    std::vector<long long> nums(n + 1, 0);
    nums[1] = 1;

    long long ans = 1;
    for (int i = 1; i <= n / 2; ++i) {
        nums[2 * i] = nums[i];

        if (2 * i + 1 <= n) {
            nums[2 * i + 1] = nums[i] + nums[i + 1];
            ans = std::max(ans, nums[2 * i + 1]);
        }
    }

    return ans;
}

// 1137. N-th Tribonacci Number
int tribo(int n, std::vector<int> &dp) {
    if (dp[n] != -1) return dp[n];

    return dp[n] = tribo(n - 1, dp) + tribo(n - 2, dp) + tribo(n - 3, dp);
}

int tribonacci(int n) {
    if (n <= 0) return 0;

    if (n <= 2) return 1;

    std::vector<int> dp(n + 1, -1);
    dp[0] = 0;
    dp[1] = 1;
    dp[2] = 1;

    return tribo(n, dp);
}

// 300. Longest Increasing Subsequence
int lengthOfLIS(std::vector<int> &nums) {
    int n = nums.size();

    std::vector<int> dp(n + 1, INF);
    dp[0] = -INF;

    int ans = 1;

    for (int i = 0; i < n; ++i) {
        int j = std::lower_bound(dp.begin(), dp.end(), nums[i]) - dp.begin();
        if (nums[i] > dp[j - 1] && nums[i] < dp[j]) {
            dp[j] = nums[i];
            ans = std::max(ans, j);
        }
    }

    return ans;
}

// 322. Coin Change

const int MAX_AMT = 1e4 + 1;

int coinChange(std::vector<int> &coins, int amount) {
    std::vector<int> dp(MAX_AMT, INF);

    dp[0] = 0;
    for (int i = 0; i < amount; ++i) {
        for (auto c: coins) {
            if ((long long) i + c >= MAX_AMT) continue;
            dp[i + c] = std::min(dp[i + c], dp[i] + 1);
        }
    }

    return dp[amount] != INF ? dp[amount] : -1;
}

// 1043. Partition Array for Maximum Sum

int maxSumAfterPartitioning(std::vector<int> &arr, int k) {
    int n = arr.size();

    std::vector<int> dp(n + 1, 0);

    for (int i = 1; i <= n; ++i) {
        int curr_max = arr[i - 1];

        for (int j = 1; j <= k && i - j >= 0; ++j) {
            curr_max = std::max(curr_max, arr[i - j]);
            dp[i] = std::max(dp[i], dp[i - j] + curr_max * j);
        }
    }

    return dp[n];
}

// 1884. Egg Drop With 2 Eggs and N Floors

int dfs(int f, int e, std::vector<std::vector<int>> &dp) {
    if (e == 1) return f;

    if (f == 0 || f == 1) return f;

    if (dp[f][e] != -1) return dp[f][e];

    dp[f][e] = INF;

    for (int i = 1; i <= f; ++i) {
        dp[f][e] = std::min(dp[f][e],
                            std::max(
                                    dfs(i - 1, e - 1, dp),
                                    dfs(f - i, e, dp)
                            ) + 1
        );
    }

    return dp[f][e];
}

int twoEggDrop(int n) {
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(2 + 1, -1));

    return dfs(n, 2, dp);
}

// 1638. Count Substrings That Differ by One Character

int countSubstrings(std::string s, std::string t) {
    int n = s.length();
    int m = t.length();

    std::vector<std::vector<int>> match(n + 1, std::vector<int>(m + 1, 0));
    std::vector<std::vector<int>> missmatch(n + 1, std::vector<int>(m + 1, 0));

    int ans = 0;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (s[i - 1] == t[j - 1]) {
                match[i][j] = match[i - 1][j - 1] + 1;
                missmatch[i][j] = missmatch[i - 1][j - 1];
            } else {
                missmatch[i][j] = 1 + match[i - 1][j - 1];
            }

            ans += missmatch[i][j];
        }
    }

    return ans;
}

int backtrack(int i, std::vector<int>& cookies, std::vector<int>& children, int k, int zero_count) {
    if (cookies.size() - i < zero_count) return INF;

    if (i == cookies.size()) {
        return *std::max_element(children.begin(), children.end());
    }

    int ans = INF;

    for (int j = 0; j < k; ++j) {
        zero_count -= children[j] == 0 ? 1 : 0;
        children[j] += cookies[i];

        ans = std::min(ans, backtrack(i + 1, cookies, children, k, zero_count));

        children[j] -= cookies[i];
        zero_count += children[j] == 0 ? 1 : 0;
    }

    return ans;
}

int distributeCookies(std::vector<int>& cookies, int k) {
    std::vector<int> children(k, 0);
    return backtrack(0, cookies, children, k, k);
}

int main() {
    auto v = std::vector<int>{1, 2};
    distributeCookies(v, 2);
    return 0;
}