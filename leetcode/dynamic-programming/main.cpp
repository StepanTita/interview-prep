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
int rob(std::vector<int>& nums) {
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

bool wordBreak(std::string s, std::vector<std::string>& wordDict) {
    TrieNode *root = new TrieNode();

    for (auto &w : wordDict) {
        buildTrie(w, root);
    }

    std::vector<int> dp(s.length(), -1);

    return dfs(s, 0, root, root, dp);
}


int main() {
    auto v = std::vector<std::string>{"a", "b", "bbb", "bbbb"};
    wordBreak("bb", v);
    return 0;
}