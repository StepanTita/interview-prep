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

// 2305. Fair Distribution of Cookies

int backtrack(int i, std::vector<int> &cookies, std::vector<int> &children, int k, int zero_count) {
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

int distributeCookies(std::vector<int> &cookies, int k) {
    std::vector<int> children(k, 0);
    return backtrack(0, cookies, children, k, k);
}

// 1140. Stone Game II

int alice(std::vector<int> &piles, std::vector<int> prefix, int l, int M, int total_sum);

std::vector<std::vector<int>> dp_alice;
std::vector<std::vector<int>> dp_bob;

int bob(std::vector<int> &piles, std::vector<int> prefix, int l, int M, int total_sum) {
    if (l >= piles.size()) return 0;

    if (dp_bob[l][M] != -1) {
        return dp_bob[l][M];
    }

    int n = piles.size();

    int max_bob = 0;
    for (int x = 1; x <= 2 * M; ++x) {
        if (l + x >= prefix.size()) break;

        // max for bob is total - max for alice
        int max_alice = alice(
                piles,
                prefix,
                l + x,
                std::max(M, x),
                total_sum - (prefix[l + x] - prefix[l])
        );

        max_bob = std::max(
                max_bob,
                total_sum - max_alice
        );
    }

    return dp_bob[l][M] = max_bob;
}

int alice(std::vector<int> &piles, std::vector<int> prefix, int l, int M, int total_sum) {
    if (l >= piles.size()) return 0;

    if (dp_alice[l][M] != -1) {
        return dp_alice[l][M];
    }

    int n = piles.size();

    int max_alice = 0;
    for (int x = 1; x <= 2 * M; ++x) {
        if (l + x >= prefix.size()) break;

        // max for alice is total - max for bob
        int max_bob = bob(
                piles,
                prefix,
                l + x,
                std::max(M, x),
                total_sum - (prefix[l + x] - prefix[l])
        );

        max_alice = std::max(
                max_alice,
                total_sum - max_bob
        );
    }

    return dp_alice[l][M] = max_alice;
}

int stoneGameII(std::vector<int> &piles) {
    int n = piles.size();

    dp_alice = std::vector<std::vector<int>>(n + 1, std::vector<int>(n + 1, -1));
    dp_bob = std::vector<std::vector<int>>(n + 1, std::vector<int>(n + 1, -1));

    std::vector<int> prefix(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        prefix[i] = prefix[i - 1] + piles[i - 1];
    }

    return alice(piles, prefix, 0, 1, prefix[n]);
}

// 647. Palindromic Substrings

int countSubstrings(std::string s) {
    int n = s.length();

    std::vector<std::vector<bool>> dp(n + 1, std::vector<bool>(n + 1, false));

    int res = 0;
    for (int k = 0; k < n; ++k) {
        for (int i = 1; i + k <= n; ++i) {
            int j = i + k;

            if (k == 0) {
                dp[i][j] = true;
                ++res;
            } else if (s[i - 1] == s[j - 1] && (k == 1 || dp[i + 1][j - 1])) {
                dp[i][j] = true;
                ++res;
            }
        }
    }

    return res;
}

// 1372. Longest ZigZag Path in a Binary Tree

int maxZigZag(TreeNode *curr, bool left, int depth = 0) {
    // left is true if came to curr from left,
    // otherwise false

    if (curr == NULL) return depth;

    if (left) {
        return std::max(
                maxZigZag(curr->right, !left, depth + 1),
                maxZigZag(curr->left, true, 0)
        );
    };
    return std::max(
            maxZigZag(curr->left, !left, depth + 1),
            maxZigZag(curr->right, false, 0)
    );
}

int longestZigZag(TreeNode *curr) {
    if (curr == NULL) {
        return 0;
    }

    return std::max(
            maxZigZag(curr->left, true),
            maxZigZag(curr->right, false)
    );
}

// 1493. Longest Subarray of 1's After Deleting One Element

int longestSubarray(std::vector<int> &nums) {
    int n = nums.size();

    int l = 0;
    int zeros = 0;
    int len = 0;
    for (int r = 0; r < n; ++r) {
        if (nums[r] == 0) ++zeros;

        while (zeros > 1 && l < r) {
            if (nums[l] == 0) --zeros;
            ++l;
        }

        len = std::max(len, r - l);
    }

    return len;
}

// 131. Palindrome Partitioning

void dfs(
        int start,
        std::string &s,
        std::vector<std::vector<bool>> &dp,
        std::vector<std::string> &part,
        std::vector<std::vector<std::string>> &res
) {
    int n = s.length();

    if (start > n) {
        res.emplace_back(part.begin(), part.end());
        return;
    }

    std::string curr = "";
    for (int j = start; j <= n; ++j) {
        curr += s[j - 1];

        if (dp[start][j]) {
            part.emplace_back(curr);
            dfs(j + 1, s, dp, part, res);
            part.pop_back();
        }
    }
}

std::vector<std::vector<std::string>> partition(std::string s) {
    int n = s.length();

    std::vector<std::vector<bool>> dp(n + 1, std::vector<bool>(n + 1, 0));

    for (int k = 0; k < n; ++k) {
        for (int i = 1; i + k <= n; ++i) {
            int j = i + k;

            if (k == 0) {
                dp[i][j] = true;
            } else if (s[i - 1] == s[j - 1] && (k == 1 || dp[i + 1][j - 1])) {
                dp[i][j] = true;
            }
        }
    }

    std::vector<std::string> curr;
    std::vector<std::vector<std::string>> res;

    dfs(1, s, dp, curr, res);

    return res;
}

// 2304. Minimum Path Cost in a Grid

int minPathCost(std::vector<std::vector<int>> &grid, std::vector<std::vector<int>> &moveCost) {
    int n = grid.size();
    int m = grid[0].size();
    std::vector<std::vector<int>> dp(2, std::vector<int>(m, INF));

    for (int j = 0; j < m; ++j) {
        dp[0][j] = grid[0][j];
    }

    // take dp to k-th column for the previous step, and add moveCost from prev step to current step
    // dp[i][j] = min(dp[i - 1][j], dp[i - 1][k] + moveCost[grid[i - 1][k]][j]);

    int ans = INF;
    for (int i = 1; i < n; ++i) {
        dp[i % 2].assign(m, INF);
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < m; ++k) {
                dp[i % 2][j] = std::min(
                        dp[i % 2][j],
                        dp[(i + 1) % 2][k] + moveCost[grid[i - 1][k]][j] + grid[i][j]
                );
            }
            if (i + 1 == n) {
                ans = std::min(ans, dp[i % 2][j]);
            }
        }
    }

    return ans;
}

// 931. Minimum Falling Path Sum

int minFallingPathSum(std::vector<std::vector<int>> &matrix) {
    int n = matrix.size();
    std::vector<std::vector<int>> dp(2, std::vector<int>(n, INF));

    if (n == 1) return matrix[0][0];

    int ans = INF;

    for (int j = 0; j < n; ++j) {
        dp[0][j] = matrix[0][j];
    }

    for (int i = 1; i < n; ++i) {
        dp[i % 2].assign(n, INF);

        for (int j = 0; j < n; ++j) {
            int topLeft = INF;
            if (j - 1 >= 0) {
                topLeft = dp[(i + 1) % 2][j - 1];
            }

            int topRight = INF;
            if (j + 1 < n) {
                topRight = dp[(i + 1) % 2][j + 1];
            }

            dp[i % 2][j] = std::min(
                    dp[i % 2][j],
                    std::min(std::min(dp[(i + 1) % 2][j], topLeft), topRight) + matrix[i][j]
            );

            if (i + 1 == n) {
                ans = std::min(ans, dp[i % 2][j]);
            }
        }
    }

    return ans;
}

// 983. Minimum Cost For Tickets

int mincostTickets(std::vector<int> &days, std::vector<int> &costs) {
    int n = days.size();

    std::vector<int> dp(days[n - 1] + 1, 0);

    int i = 0;

    for (int d = days[0]; d <= days[n - 1]; ++d) {
        if (d < days[i]) {
            dp[d] = dp[d - 1];
        } else {
            dp[d] = std::min(
                    dp[std::max(d - 30, 0)] + costs[2],
                    std::min(dp[d - 1] + costs[0], dp[std::max(d - 7, 0)] + costs[1])
            );
            ++i;
        }
    }

    return dp[days[n - 1]];
}

// 413. Arithmetic Slices

int numberOfArithmeticSlices(std::vector<int> &nums) {
    int n = nums.size();

    if (n < 3) return 0;

    std::vector<std::vector<int>> dp(n, std::vector<int>(n, INF));

    int res = 0;
    for (int i = 1; i < n - 1; ++i) {
        if (nums[i] - nums[i - 1] == nums[i + 1] - nums[i]) {
            dp[i - 1][i + 1] = nums[i] - nums[i - 1];
            ++res;
        }
    }

    for (int k = 4; k <= n; ++k) {
        for (int i = 0; i + k - 1 < n; ++i) {
            int j = i + k - 1;

            if (dp[i][j - 1] == nums[j] - nums[j - 1]) {
                dp[i][j] = dp[i][j - 1];
                ++res;
            }
        }
    }

    return res;
}

// 712. Minimum ASCII Delete Sum for Two Strings

int minimumDeleteSum(std::string s1, std::string s2) {
    int n = s1.size();
    int m = s2.size();

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));

    for (int i = n - 1; i >= 0; --i) {
        for (int j = m - 1; j >= 0; --j) {
            if (s1[i] == s2[j]) {
                dp[i][j] = dp[i + 1][j + 1] + s1[i];
            } else {
                dp[i][j] = std::max(dp[i + 1][j], dp[i][j + 1]);
            }
        }
    }
    int lcsS1 = 0;
    for (int i = 0; i < n; ++i) {
        lcsS1 += (int) s1[i];
    }

    int lcsS2 = 0;
    for (int j = 0; j < m; ++j) {
        lcsS2 += (int) s2[j];
    }

    return lcsS1 + lcsS2 - 2 * dp[0][0];
}

// 241. Different Ways to Add Parentheses

bool isDigit(char c) {
    return c >= '0' && c <= '9';
}

int apply(int a, int b, char op) {
    if (op == '+') {
        return a + b;
    } else if (op == '-') {
        return a - b;
    } else if (op == '*') {
        return a * b;
    }

    return -INF;
}

int evaluate(std::string expr, int l, int r) {
    std::stack<int> st;
    st.push(0);

    char op = '#';
    for (int i = l; i <= r; ++i) {
        if (!isDigit(expr[i])) {
            op = expr[i];
            st.push(0);
        } else {
            int v = st.top();
            st.pop();
            v = v * 10 + (expr[i] - '0');
            st.push(v);
        }
    }

    int a = st.top();
    st.pop();
    int b = st.top();

    return apply(b, a, op);
}

std::vector<int> dfs(std::string expression, int l, int r, int ops, std::vector<std::vector<std::vector<int>>> &dp) {
    if (ops == 0) {
        return std::vector<int>{std::stoi(std::string(expression.begin() + l, expression.begin() + r + 1))};
    }
    if (ops == 1) {
        return std::vector<int>{evaluate(expression, l, r)};
    }

    int leftOps = 0;

    std::vector<int> res;
    for (int i = l; i <= r; ++i) {
        if (!isDigit(expression[i])) {
            auto left = dfs(expression, l, i - 1, leftOps, dp);
            auto right = dfs(expression, i + 1, r, ops - leftOps - 1, dp);

            for (int li: left) {
                for (int ri: right) {
                    res.emplace_back(apply(li, ri, expression[i]));
                }
            }

            ++leftOps;
        }
    }

    return dp[l][r] = res;
}

std::vector<int> diffWaysToCompute(std::string expression) {
    int n = expression.length();

    auto dp = std::vector<std::vector<std::vector<int>>>(n, std::vector<std::vector<int>>(n, std::vector<int>()));

    int ops = 0;
    for (char c: expression) {
        if (!isDigit(c)) {
            ++ops;
        }
    }

    if (ops == 0) return std::vector<int>{std::stoi(expression)};

    return dfs(expression, 0, n - 1, ops, dp);
}

// 526. Beautiful Arrangement

bool bitSet(int n, int i) {
    return n & (1 << i);
}

int setBit(int n, int i) {
    return n | (1 << i);
}

int dfs(int n, int i, int used, std::vector<std::vector<int>> &dp) {
    if (i > n) return 1;

    if (dp[i][used] != -1) return dp[i][used];

    int ans = 0;

    for (int j = 1; j <= n; ++j) {
        if (bitSet(used, j)) continue;

        if (i % j == 0 || j % i == 0) {
            ans += dfs(n, i + 1, setBit(used, j), dp);
        }
    }
    return dp[i][used] = ans;
}

int countArrangement(int n) {
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>((1 << 16) + 1, -1));
    return dfs(n, 1, 0, dp);
}

// 1664. Ways to Make a Fair Array

int waysToMakeFair(std::vector<int> &nums) {
    int n = nums.size();

    int prefix_odd = 0;
    int prefix_even = 0;

    int count = 0;

    std::vector<int> suffix_odd(n + 1, 0);
    std::vector<int> suffix_even(n + 1, 0);
    for (int i = n - 1; i >= 0; --i) {
        if (i % 2 != 0) {
            suffix_odd[i] = suffix_odd[i + 1] + nums[i];
            suffix_even[i] = suffix_even[i + 1];
        } else {
            suffix_even[i] = suffix_even[i + 1] + nums[i];
            suffix_odd[i] = suffix_odd[i + 1];
        }
    }

    for (int i = 0; i < n; ++i) {
        int suff_odd = suffix_even[i + 1];
        int suff_even = suffix_odd[i + 1];
        if (prefix_odd + suff_odd == prefix_even + suff_even) {
            ++count;
        }

        if (i % 2 != 0) {
            prefix_odd += nums[i];
        } else {
            prefix_even += nums[i];
        }
    }

    return count;
}

// 518. Coin Change II

int change(int amount, std::vector<int> &coins) {
    int n = coins.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(amount + 1, 0));

    for (int i = 1; i <= n; ++i) {
        dp[i][0] = 1;
        for (int j = 1; j <= amount; ++j) {
            dp[i][j] = dp[i - 1][j] + ((j - coins[i - 1] >= 0) ? dp[i][j - coins[i - 1]] : 0);
        }
    }

    return dp[n][amount];
}

// 1035. Uncrossed Lines

int maxUncrossedLines(std::vector<int> &nums1, std::vector<int> &nums2) {
    int n = nums1.size();
    int m = nums2.size();

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (nums1[i - 1] == nums2[j - 1]) {
                dp[i][j] = std::max(dp[i][j], dp[i - 1][j - 1] + 1);
            } else {
                dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[n][m];
}

// 516. Longest Palindromic Subsequence

int longestPalindromeSubseq(std::string s) {
    int n = s.length();

    std::vector<int> dp(n, 0);
    std::vector<int> dp_prev(n, 0);

    for (int i = n - 1; i >= 0; --i) {
        dp[i] = 1;

        for (int j = i + 1; j < n; ++j) {
            if (s[i] == s[j]) {
                dp[j] = dp_prev[j - 1] + 2;
            } else {
                dp[j] = std::max(dp_prev[j], dp[j - 1]);
            }
        }

        std::swap(dp, dp_prev);
    }

    return dp_prev[n - 1];
}

// 1947. Maximum Compatibility Score Sum

int backtrack(int s, int taken,
              std::vector<std::vector<int>> &students,
              std::vector<std::vector<int>> &mentors,
              std::vector<int> &memo,
              std::vector<std::vector<int>> &compat
) {
    int m = students.size();
    int n = students[0].size();

    if (s >= m) return 0;

    if (memo[taken] != -1) return memo[taken];

    for (int i = 0; i < m; ++i) {
        if (bitSet(taken, i)) continue;

        memo[taken] = std::max(
                memo[taken],
                backtrack(s + 1, setBit(taken, i), students, mentors, memo, compat) + compat[s][i]
        );
    }

    return memo[taken];
}

int maxCompatibilitySum(std::vector<std::vector<int>> &students, std::vector<std::vector<int>> &mentors) {
    int m = students.size();
    int n = students[0].size();

    std::vector<int> memo((1 << m) + 1, -1);
    std::vector<std::vector<int>> compat(m, std::vector<int>(m, 0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < n; ++k) {
                compat[i][j] += students[i][k] == mentors[j][k];
            }
        }
    }

    return backtrack(0, 0, students, mentors, memo, compat);
}

// 926. Flip String to Monotone Increasing

int minFlipsMonoIncr(std::string s) {
    // dp[i][0] - min cost of i-th prefix given that everything before i is 0
    // dp[i][1] - min cost of i-th prefix given that not everything before i is 0

    int n = s.length();

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(2, 0));

    for (int i = 1; i <= n; ++i) {
        dp[i][0] = dp[i - 1][0] + (s[i - 1] != '0');
        dp[i][1] = std::min(dp[i - 1][0], dp[i - 1][1]) + (s[i - 1] == '0');
    }

    return std::min(dp[n][0], dp[n][1]);
}

// 583. Delete Operation for Two Strings

int lcs(std::string &word1, std::string &word2) {
    int n = word1.size();
    int m = word2.size();

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));

    for (int i = n - 1; i >= 0; --i) {
        for (int j = m - 1; j >= 0; --j) {
            if (word1[i] == word2[j]) {
                dp[i][j] = dp[i + 1][j + 1] + 1;
            } else {
                dp[i][j] = std::max(dp[i + 1][j], dp[i][j + 1]);
            }
        }
    }

    return dp[0][0];
}

int minDistance(std::string word1, std::string word2) {
    int n = word1.size();
    int m = word2.size();

    int len = lcs(word1, word2);
    return n + m - 2 * len;
}

// 1048. Longest String Chain

bool isPred(std::string &s, std::string &t) {
    if (t.length() - s.length() != 1) return false;

    int i = 0;
    int j = 0;
    int counter = 1;
    while (i < s.length() && j < t.length()) {
        if (s[i] == t[j]) {
            ++i;
            ++j;
        } else if (counter == 1) {
            counter = 0;
            ++j;
        } else {
            return false;
        }
    }

    return true;
}

int dfs(std::string &curr,
        std::unordered_map<std::string, std::unordered_set<std::string> > &pred,
        std::unordered_set<std::string> &visited,
        std::unordered_map<std::string, int> &memo
) {
    if (memo.contains(curr)) return memo[curr];

    visited.insert(curr);

    int ans = 0;

    for (auto next: pred[curr]) {
        if (!visited.contains(next)) {
            ans = std::max(ans, dfs(next, pred, visited, memo) + 1);
        }
    }

    visited.erase(curr);

    return memo[curr] = ans;
}

int longestStrChain(std::vector<std::string> &words) {
    int n = words.size();
    std::unordered_map<std::string, std::unordered_set<std::string> > pred;

    for (auto &word: words) {
        for (auto &next: words) {
            if (word == next) continue;

            if (word.length() == 1 && next.length() == 1) continue;

            if (isPred(word, next)) {
                pred[word].insert(next);
            }
        }
    }

    std::unordered_map<std::string, int> memo;
    std::unordered_set < std::string > visited;

    int ans = 1;
    for (auto &word: words) {
        ans = std::max(ans, dfs(word, pred, visited, memo) + 1);
    }

    return ans;
}

// 96. Unique Binary Search Trees

int numTrees(int n) {
    if (n < 2) return 1;

    int s = 0;

    for (int i = 0; i < n; ++i) {
        s += numTrees(i) * numTrees(n - i - 1);
    }

    return s;
}

// 343. Integer Break

int backtrack(int n, std::vector<int> &dp) {
    if (dp[n] != -1) return dp[n];

    for (int i = 1; i < n; ++i) {
        dp[n] = std::max(dp[n], (n - i) * i);
        dp[n] = std::max(dp[n], backtrack(n - i, dp) * i);
    }

    return dp[n];
}

int integerBreak(int n) {
    std::vector<int> dp(n + 1, -1);

    return backtrack(n, dp);
}

// 1031. Maximum Sum of Two Non-Overlapping Subarrays

int maxSum(std::vector<int> &p, int L, int M) {
    int n = p.size();

    int ans = 0;
    int max_left = 0;
    for (int i = L + M; i < n; ++i) {
        max_left = std::max(max_left, p[i - M] - p[i - M - L]);
        ans = std::max(ans, max_left + p[i] - p[i - M]);
    }

    return ans;
}

int maxSumTwoNoOverlap(std::vector<int> &nums, int firstLen, int secondLen) {
    int n = nums.size();
    std::vector<int> prefix(n + 1, 0);
    for (int i = 1; i <= n; ++i) {
        prefix[i] = prefix[i - 1] + nums[i - 1];
    }

    return std::max(
            maxSum(prefix, firstLen, secondLen),
            maxSum(prefix, secondLen, firstLen)
    );
}

// 1105. Filling Bookcase Shelves

int minHeightShelves(std::vector<std::vector<int>> &books, int shelfWidth) {
    int n = books.size();

    std::vector<int> dp(n + 1, 0);

    dp[0] = 0;

    for (int i = 0; i < n; ++i) {
        dp[i + 1] = dp[i] + books[i][1];

        int sum_w = 0;
        int max_h = 0;
        for (int j = i; j >= 0; --j) {
            sum_w += books[j][0];

            if (sum_w > shelfWidth) break;

            max_h = std::max(max_h, books[j][1]);
            dp[i + 1] = std::min(dp[i + 1], dp[j] + max_h);
        }
    }

    return dp[n];
}

// 1014. Best Sightseeing Pair

int maxScoreSightseeingPair(std::vector<int> &values) {
    int prev_i = 0;

    int ans = 0;
    for (int j = 1; j < values.size(); ++j) {
        ans = std::max(ans, values[prev_i] + prev_i + values[j] - j);

        if (j + values[j] > prev_i + values[prev_i]) {
            prev_i = j;
        }
    }

    return ans;
}

// 1911. Maximum Alternating Subsequence Sum

long long maxAlternatingSum(std::vector<int> &nums) {
    int n = nums.size();

    long long even = 0;
    long long odd = 0;

    for (int i = 0; i < n; ++i) {
        even = std::max(even, odd + nums[i]);
        odd = std::max(odd, even - nums[i]);
    }

    return even;
}

// 1749. Maximum Absolute Sum of Any Subarray

int maxAbsoluteSum(std::vector<int> &nums) {
    int n = nums.size();

    int max_sum = 0;
    int min_sum = 0;

    int ans = 0;

    for (int i = 0; i < n; ++i) {
        max_sum = std::max(max_sum + nums[i], nums[i]);
        min_sum = std::min(min_sum + nums[i], nums[i]);

        if (max_sum < 0) {
            max_sum = 0;
        }
        if (min_sum > 0) {
            min_sum = 0;
        }

        ans = std::max(ans, std::max(std::abs(max_sum), std::abs(min_sum)));
    }

    return ans;
}

// 1653. Minimum Deletions to Make String Balanced

int minimumDeletions(std::string s) {
    int n = s.length();

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(2, 0));

    // dp[i][0] - currently in the first part
    // dp[i][1] - currently in the second part

    int a = 0;
    int b = 0;
    for (int i = 1; i <= n; ++i) {
        b = std::min(a, b) + (s[i - 1] == 'a');
        a = a + (s[i - 1] == 'b');
    }

    return std::min(a, b);
}

// 2110. Number of Smooth Descent Periods of a Stock

long long getDescentPeriods(std::vector<int> &prices) {
    // dp[i] - number of smooth descent periods ending in i-th price
    // dp[i] = dp[i - 1] + 1 if prices[i] - prices[i - 1] == 1
    // else dp[i] = 1

    int n = prices.size();

    long long ans = 1;
    long long sum = 1;

    for (int i = 2; i <= n; ++i) {
        if ((prices[i - 2] - prices[i - 1]) == 1) {
            ++sum;
        } else {
            sum = 1;
        }
        ans += sum;
    }

    return ans;
}

// 2673. Make Costs of Paths Equal in a Binary Tree

int left(std::vector<int> &tree, int i) {
    if (2 * i + 1 >= tree.size()) return 0;
    return tree[2 * i + 1];
}

int right(std::vector<int> &tree, int i) {
    if (2 * i + 2 >= tree.size()) return 0;
    return tree[2 * i + 2];
}

int minIncrements(int n, std::vector<int> &cost) {
    int ans = 0;
    for (int i = n - 1; i >= 0; --i) {
        int diff = std::max(left(cost, i), right(cost, i)) -
                   std::min(left(cost, i), right(cost, i));
        ans += diff;

        cost[i] += std::max(left(cost, i), right(cost, i));
    }

    return ans;
}

// 1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance

std::vector<std::vector<int>> findClosure(int n, std::vector<std::vector<int>> adjMatrix, int distanceThreshold) {
    std::vector<std::vector<int>> closure(n, std::vector<int>(n, 0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            closure[i][j] = adjMatrix[i][j] <= distanceThreshold ? adjMatrix[i][j] : INF;
        }
    }

    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (closure[i][k] != INF && closure[k][j] != INF &&
                    closure[i][k] + closure[k][j] <= distanceThreshold) {
                    closure[i][j] = std::min(closure[i][j], closure[i][k] + closure[k][j]);
                }
            }
        }
    }

    return closure;
}

int findTheCity(int n, std::vector<std::vector<int>> &edges, int distanceThreshold) {
    std::vector<std::vector<int>> adjMatrix(n, std::vector<int>(n, INF));

    for (auto &e: edges) {
        adjMatrix[e[0]][e[1]] = e[2];
        adjMatrix[e[1]][e[0]] = e[2];
    }

    auto closure = findClosure(n, adjMatrix, distanceThreshold);

    int minNeighbors = n + 1;
    int ans = -1;
    for (int i = n - 1; i >= 0; --i) {
        int count = 0;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            count += (closure[i][j] != INF);
        }

        if (minNeighbors > count) {
            minNeighbors = count;
            ans = i;
        }
    }

    return ans;
}

// 1690. Stone Game VII

int stoneGameVII(std::vector<int> &stones) {
    int n = stones.size();

    std::vector<int> prefix(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        prefix[i + 1] = prefix[i] + stones[i];
    }

    std::vector<std::vector<int>> dp(n, std::vector<int>(n, 0));

    for (int l = n - 1; l >= 0; --l) {
        for (int r = l + 1; r < n; ++r) {
            dp[l][r] = std::max(
                    (prefix[r + 1] - prefix[l + 1]) - dp[l + 1][r],
                    (prefix[r] - prefix[l]) - dp[l][r - 1]
            );
        }
    }

    return dp[0][n - 1];
}

// 143. Longest Common Subsequence

int longestCommonSubsequence(std::string a, std::string b) {
    int n = a.size();
    int m = b.size();

    std::vector<int> dp(m + 1, 0);

    int ans = 0;

    for (int i = 1; i <= n; ++i) {
        int prev_val = 0;
        for (int j = 1; j <= m; ++j) {
            // dp[j] - is dp[i - 1][j]
            // dp[j - 1] - is dp[i][j - 1]
            // prev_val - is dp[i - 1][j - 1]

            int curr_val = dp[j]; // dp[i - 1][j]

            if (a[i - 1] == b[j - 1]) {
                dp[j] = prev_val + 1;
            } else {
                dp[j] = std::max(dp[j], dp[j - 1]);
            }

            prev_val = curr_val; // j + 1 -> dp[i - 1][j - 1]
            ans = std::max(ans, dp[j]);
        }
    }

    return ans;
}

// 1092. Shortest Common Supersequence

std::vector<std::vector<int>> lcs(const std::string &a, const std::string &b) {
    int n = a.length();
    int m = b.length();

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = std::max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }

    return dp;
}

std::string shortestCommonSupersequence(const std::string a, const std::string b) {
    int n = a.length();
    int m = b.length();

    auto lcaDP = lcs(a, b);

    std::string res = "";

    int i = n;
    int j = m;
    while (i > 0 && j > 0) {
        if (a[i - 1] == b[j - 1]) {
            res += a[i];
            --i;
            --j;
        } else {
            if (lcaDP[i - 1][j] > lcaDP[i][j - 1]) {
                res += a[i - 1];
                --i;
            } else {
                res += b[j - 1];
                --j;
            }
        }
    }

    while (i > 0) {
        res += a[i--];
    }

    while (j > 0) {
        res += b[j--];
    }

    return res;
}

// 121. Best Time to Buy and Sell Stock

int maxProfit(std::vector<int> &prices) {
    int max_profit = 0;
    int hold = prices[0];

    /*
    We have some stock 'hold', and we are at index i of prices.
    If we could have sold 'hold' better than in the future, we would have tried that,
    Otherwise right now prices[i] < hold, so we will defenitely not get any profit
    from selling bigger hold in the future.
    */
    for (int i = 0; i < prices.size(); ++i) {
        if (hold > prices[i]) {
            hold = prices[i];
        }
        max_profit = std::max(max_profit, prices[i] - hold);
    }

    return max_profit;
}

// 122. Best Time to Buy and Sell Stock II

int maxProfit2(std::vector<int> &prices) {
    int n = prices.size();

    // dp[i][0] - max profit not holding stock
    // dp[i][1] - stock that we are holding

    /*
     * any given moment we have only 3 options,
     * either sell our current stock that we hold,
     * buy a new stock, or just keep what we have
     * for every prices i, we can consider all of the following:
     * if prices[i] - hold > 0 - means current sell is profitable - do it
     * after that immediately buy a current stock
     * if there is a more profitable deal prices[j] for the prev stock somewhere in the future,
     * then it cannot be more profitable than:
     *  - selling prev stock (prices[i] - hold)
     *  - buying prices[i] stock (we are already in profit here
     *  - selling prices[i] for prices[j]
     *  so if prices[j] - hold > prices[i] - hold (j > i)
     *  then still: prices[j] - hold <= (prices[i] - hold) + (prices[j] - prices[i])
     *  moreover prices[j] - hold == (prices[i] - hold) + (prices[j] - prices[i]) (simplify the right part)
     */
    int max_profit = 0;
    int hold = prices[0];
    for (int i = 0; i < n; ++i) {
        if (prices[i] - hold > 0) {
            max_profit += (prices[i] - hold);
            hold = prices[i];
        }
        hold = std::min(hold, prices[i]);
    }

    return max_profit;
}

int maxProfitWithCooldown(std::vector<int> &prices) {
    // on each day i, we can either buy stock
    // or sell stock, or we have a cooldown
    // dp[i][0] - no stock
    // dp[i][1] - holding
    // dp[i][2] - cooldown

    // for the no stock case we might have sold and experienced a cooldown, or have not done anything
    // dp[i][0] = max(dp[i - 1][0], dp[i - 1][2]);
    // for the holding case we might have been holding it for a while, or had nothing and bought
    // dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
    // for cooldown only one option possible: we had something and we have sold it
    // dp[i][2] = dp[i - 1][1] + prices[i];

    int n = prices.size();

    int no_stock = 0;
    int hold = -prices[0];
    int cooldown = 0;
    for (int i = 1; i < n; ++i) {
        int prev_no_stock = no_stock;
        no_stock = std::max(no_stock, cooldown);

        int prev_hold = hold;
        hold = std::max(hold, prev_no_stock - prices[i]);

        cooldown = prev_hold + prices[i];
    }

    return std::max(no_stock, cooldown);
}

// 123. Best Time to Buy and Sell Stock III

int maxProfit2Transactions(std::vector<int> &prices) {
    // every time at price i, I have either completed 0 transactions
    // or I have completed 1 transaction
    // or I have completed 2 transactions
    // also for each of those cases except for the last one I may or may not have
    // a stock that I hold
    // dp[i][0] - completed 0 transactions
    // dp[i][1] - completed 1 transaction
    // dp[i][2] - completed 2 transactions

    // dp[i][t][0] - not holding any stock after t transactions
    // dp[i][t][1] - holding a stock after t transactions

    // either we complete transaction if we hold something, or stay with what we have
    // dp[i][t][0] = max(dp[i - 1][t][0], dp[i - 1][t - 1][1] + prices[i]);
    // we either stay in a current transaction, or engage in a new transaction
    // here we use t - 1 because we do not need to calculate dp[i][2][1], but need dp[i][0][1]
    // dp[i][t - 1][1] = max(dp[i - 1][t - 1][1], dp[i - 1][t - 1][0] - prices[i]);

    int n = prices.size();

    int k = 2;

    std::vector<std::vector<std::vector<int>>> dp(n, std::vector<std::vector<int>>(k + 1, std::vector<int>(2, -INF)));

    dp[0][0][0] = 0;
    dp[0][0][1] = -prices[0];

    for (int i = 1; i < n; ++i) {
        dp[i][0][0] = 0;
        for (int t = 1; t <= k; ++t) {
            dp[i][t][0] = std::max(dp[i - 1][t][0], dp[i - 1][t - 1][1] + prices[i]);

            dp[i][t - 1][1] = std::max(dp[i - 1][t - 1][1], dp[i - 1][t - 1][0] - prices[i]);
        }
    }

    return std::max({dp[n - 1][0][0], dp[n - 1][1][0], dp[n - 1][2][0]});
}

// Best Time to Buy and Sell Stock IV

int maxProfit(int k, std::vector<int> &prices) {
    // every time at price i, I have either completed 0 transactions
    // or I have completed 1 transaction
    // or I have completed 2 transactions
    // also for each of those cases except for the last one I may or may not have
    // a stock that I hold
    // dp[i][0] - completed 0 transactions
    // dp[i][1] - completed 1 transaction
    // dp[i][2] - completed 2 transactions

    // dp[i][t][0] - not holding any stock after t transactions
    // dp[i][t][1] - holding a stock after t transactions

    // either we complete transaction if we hold something, or stay with what we have
    // dp[i][t][0] = max(dp[i - 1][t][0], dp[i - 1][t - 1][1] + prices[i]);
    // we either stay in a current transaction, or engage in a new transaction
    // here we use t - 1 because we do not need to calculate dp[i][2][1], but need dp[i][0][1]
    // dp[i][t - 1][1] = max(dp[i - 1][t - 1][1], dp[i - 1][t - 1][0] - prices[i]);

    int n = prices.size();

    std::vector<std::vector<std::vector<int>>> dp(n, std::vector<std::vector<int>>(k + 1, std::vector<int>(2, -INF)));

    dp[0][0][0] = 0;
    dp[0][0][1] = -prices[0];

    for (int i = 1; i < n; ++i) {
        dp[i][0][0] = 0;
        for (int t = 1; t <= k; ++t) {
            dp[i][t][0] = std::max(dp[i - 1][t][0], dp[i - 1][t - 1][1] + prices[i]);

            dp[i][t - 1][1] = std::max(dp[i - 1][t - 1][1], dp[i - 1][t - 1][0] - prices[i]);
        }
    }

    int ans = 0;
    for (int t = 0; t <= k; ++t) {
        ans = std::max(ans, dp[n - 1][t][0]);
    }

    return ans;
}

// 72. Edit Distance

int minEditDistance(std::string a, std::string b) {
    int n = a.length();
    int m = b.length();

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));

    for (int j = 1; j <= m; ++j) {
        dp[0][j] = dp[0][j - 1] + 1;
    }

    for (int i = 1; i <= n; ++i) {
        dp[i][0] = dp[i - 1][0] + 1;

        for (int j = 1; j <= m; ++j) {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]}) + 1;
            }
        }
    }

    return dp[n][m];
}

// 2002. Maximum Product of the Length of Two Palindromic Subsequences

bool isPalindrome(std::string &s, int idxs) {
    int n = s.length();

    std::string cand = "";
    for (int i = 0; i < n; ++i) {
        if (bitSet(idxs, i)) {
            cand += s[i];
        }
    }

    int k = cand.length();
    int m = k / 2;

    if (k % 2 == 0) {
        for (int t = 0; m - t - 1 >= 0 && m + t < k; ++t) {
            if (cand[m - t - 1] != cand[m + t]) return false;
        }
    } else {
        for (int t = 0; m - t >= 0 && m + t < k; ++t) {
            if (cand[m - t] != cand[m + t]) return false;
        }
    }

    return true;
}

int countBits(int n) {
    int count = 0;
    while (n) {
        count += n & 1;
        n >>= 1;
    }
    return count;
}

int dfs(std::string &s, int curr, int idxs, int used) {
    int ans = 0;

    if (isPalindrome(s, idxs) && isPalindrome(s, used ^ idxs)) {
        ans = countBits(idxs) * countBits(used ^ idxs);
    }

    for (int i = curr; i < s.length(); ++i) {
        if (bitSet(used, i)) continue;
        // use in the first string
        ans = std::max(ans, dfs(s, i + 1, setBit(idxs, i), setBit(used, i)));
        // use in the second string
        ans = std::max(ans, dfs(s, i + 1, idxs, setBit(used, i)));
    }

    return ans;
}

int maxProduct(std::string s) {
    return dfs(s, 0, 0, 0);
}

// 2606. Find the Substring With Maximum Cost

int toDig(char c) {
    return c - 'a';
}

int getCost(char c, std::vector<int> &chars) {
    if (chars[toDig(c)] != -INF) return chars[toDig(c)];
    return toDig(c) + 1;
}

int maximumCostSubstring(std::string s, std::string chars, std::vector<int> &vals) {
    int n = s.length();

    std::vector<int> charsMap('z' - 'a' + 1, -INF);

    for (int i = 0; i < chars.length(); ++i) {
        charsMap[toDig(chars[i])] = vals[i];
    }

    int ans = 0;
    int maxVal = 0;
    for (int i = 0; i < n; ++i) {
        maxVal = std::max(maxVal + getCost(s[i], charsMap), getCost(s[i], charsMap));
        maxVal = std::max(maxVal, 0);
        ans = std::max(ans, maxVal);
    }

    return ans;
}

// 95. Unique Binary Search Trees II

// n - number of nodes in the tree
// l - leftmost element of the current BST
// r - rightmost element of the current BST
std::vector<TreeNode *> backtrack(int n, int l, int r) {
    if (n <= 0) return std::vector<TreeNode *>{};

    auto res = std::vector<TreeNode *>{};
    for (int i = 0; i < n; ++i) {
        // when i == 0 -> n - 1 elements to the right, and 1 for root
        // when i == n - 1 -> n - 1 elements to the left, and 1 for root
        auto left = backtrack(i, l, l + i);
        auto right = backtrack(n - i - 1, l + i + 1, r);

        if (left.empty() && right.empty()) {
            auto root = new TreeNode(l + i);
            res.emplace_back(root);
        } else if (left.empty()) {
            for (auto rc: right) {
                auto root = new TreeNode(l + i);
                root->right = rc;
                res.emplace_back(root);
            }
        } else if (right.empty()) {
            for (auto lc: left) {
                auto root = new TreeNode(l + i);
                root->left = lc;
                res.emplace_back(root);
            }
        }

        for (auto lc: left) {
            for (auto rc: right) {
                auto root = new TreeNode(l + i);
                root->left = lc;
                root->right = rc;
                res.emplace_back(root);
            }
        }
    }

    return res;
}

std::vector<TreeNode *> generateTrees(int n) {
    return backtrack(n, 0, n - 1);
}

// 740. Delete and Earn

int deleteAndEarn(std::vector<int> &nums) {
    int n = nums.size();

    std::vector<int> values(1e4 + 1, 0);
    for (int num: nums) {
        values[num] += num;
    }

    int take = 0;
    int skip = 0;

    for (int i = 0; i < values.size(); ++i) {
        int t = std::max(take, skip + values[i]);
        skip = take;
        take = t;
    }

    return take;
}

// 486. Predict the Winner
/*
1 5 5 5 5
0 1 4 4 4
0 0 1 3 3
0 0 0 1 2
0 0 0 0 1
*/

bool predictTheWinner(std::vector<int> &nums) {
    int n = nums.size();

    std::vector<int> dp(n, 0);

    for (int i = n - 1; i >= 0; --i) {
        dp[i] = nums[i];
        for (int j = i + 1; j < n; ++j) {
            dp[j] = std::max(nums[i] - dp[j], nums[j] - dp[j - 1]);
        }
    }

    return dp[n - 1] >= 0;
}

// 2466. Count Ways To Build Good Strings

const int MOD = 1e9 + 7;

int countGoodStrings(int low, int high, int zero, int one) {
    std::vector<int> dp(high + 1, 0);

    dp[0] = 1;
    int res = 0;
    for (int i = 0; i <= high; ++i) {
        if (i + zero <= high) {
            dp[i + zero] = (dp[i + zero] + dp[i]) % MOD;
        }

        if (i + one <= high) {
            dp[i + one] = (dp[i + one] + dp[i]) % MOD;
        }

        if (i >= low && i <= high) {
            res = (res + dp[i]) % MOD;
        }
    }

    return res;
}

// 1218. Longest Arithmetic Subsequence of Given Difference

int longestSubsequence(std::vector<int> &arr, int difference) {
    int n = arr.size();

    std::unordered_map<int, int> dp;

    int ans = 0;
    for (int i = 0; i < n; ++i) {
        dp[arr[i]] = dp[arr[i] - difference] + 1;

        ans = std::max(ans, dp[arr[i]]);
    }

    return ans;
}

// 2140. Solving Questions With Brainpower

long long mostPoints(std::vector<std::vector<int>> &questions) {
    int n = questions.size();

    std::vector<long long> dp(n + 1, 0);

    for (int i = n - 1; i >= 0; --i) {
        if (i + questions[i][1] + 1 >= n + 1) {
            dp[i] = std::max(dp[i + 1], (long long) questions[i][0]);
        } else {
            dp[i] = std::max(dp[i + 1], dp[i + questions[i][1] + 1] + questions[i][0]);
        }
    }

    return dp[0];
}

// 1049. Last Stone Weight II

int dfs(int i, std::vector<int> &stones, int currSum, std::vector<std::unordered_map<int, int>> &memo) {
    if (i >= stones.size()) {
        if (currSum < 0) return INF;
        return currSum;
    }

    if (memo[i].contains(currSum)) {
        return memo[i][currSum];
    }

    return memo[i][currSum] = std::min(
            dfs(i + 1, stones, currSum - stones[i], memo),
            dfs(i + 1, stones, currSum + stones[i], memo)
    );
}

int lastStoneWeightII(std::vector<int> &stones) {
    int n = stones.size();

    std::vector<std::unordered_map<int, int>> memo = std::vector<std::unordered_map<int, int>>(n);
    return dfs(0, stones, 0, memo);
}

// 2063. Vowels of All Substrings

bool isVowel(char c) {
    if (c == 'a') return true;
    else if (c == 'e') return true;
    else if (c == 'i') return true;
    else if (c == 'o') return true;
    else if (c == 'u') return true;

    return false;
}

long long countVowels(std::string word) {
    int n = word.length();

    long long res = 0;
    for (int i = 0; i < word.length(); ++i) {
        if (isVowel(word[i])) {
            res += (long long) (i + 1) * (long long) (n - i);
        }
    }

    return res;
}

// 337. House Robber III

int dfs(TreeNode *root, std::unordered_map<TreeNode *, std::vector<int>> &memo, bool skip = false) {
    if (root == NULL) return 0;

    if (memo.contains(root) && memo[root][skip] != -1) return memo[root][skip];

    memo[root] = std::vector<int>(2, -1);

    if (skip) {
        return memo[root][skip] = dfs(root->left, memo, false) + dfs(root->right, memo, false);
    }

    return memo[root][skip] = std::max(
            dfs(root->left, memo, true) + root->val + dfs(root->right, memo, true),
            dfs(root->left, memo, false) + dfs(root->right, memo, false)
    );
}

int rob(TreeNode *root) {
    std::unordered_map<TreeNode *, std::vector<int>> memo;
    return dfs(root, memo);
}

// 213. House Robber II

int robOne(std::vector<int> &nums, int l, int r) {
    int take = 0;
    int skip = 0;

    for (int i = l; i < r; ++i) {
        int t = take;
        take = std::max(take, skip + nums[i]);
        skip = t;
    }

    return take;
}

int rob2(std::vector<int> &nums) {
    int n = nums.size();
    if (n == 1) return nums[0];
    return std::max(robOne(nums, 0, n - 1), robOne(nums, 1, n));
}

// 377. Combination Sum IV

int combinationSum4(std::vector<int> &nums, int target) {
    int n = nums.size();

    std::vector<unsigned long long> dp(target + 1, 0);
    dp[0] = 1;
    for (int t = 0; t <= target; ++t) {
        for (int i = 0; i < n; ++i) {
            if (t - nums[i] >= 0) {
                dp[t] += dp[t - nums[i]];
            }
        }
    }

    return dp[target];
}

// 813. Largest Sum of Averages

double partition(int start, std::vector<int> &nums, int k,
                 std::vector<std::vector<double>> &memo
) {
    if (k <= 0) return 0;

    int n = nums.size();

    if (start >= n) return 0;

    if (memo[start][k] != -1) return memo[start][k];

    int currSum = 0;
    int currCount = 0;

    for (int i = start; i < n; ++i) {
        currSum += nums[i];
        ++currCount;

        if (k != 1 || i == n - 1) {
            memo[start][k] = std::max(memo[start][k],
                                      ((double) currSum / (double) currCount) + partition(i + 1, nums, k - 1, memo));
        }
    }

    return memo[start][k];
}

double largestSumOfAverages(std::vector<int> &nums, int k) {
    int n = nums.size();

    std::vector<std::vector<double>> memo(n, std::vector<double>(k + 1, -1));
    return partition(0, nums, k, memo);
}

// 55. Jump Game

bool canJump(std::vector<int> &nums) {
    int maxJump = 0;
    for (int i = 0; i < nums.size() - 1; ++i) {
        if (maxJump < i) return false;
        maxJump = std::max(maxJump, i + nums[i]);
    }

    return maxJump >= nums.size() - 1;
}

// 97. Interleaving String

bool isInterleave(std::string a, std::string b, std::string t) {
    if (a.empty()) return b == t;
    else if (b.empty()) return a == t;

    int n = a.length();
    int m = b.length();

    if (n + m != t.length()) return false;

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));

    dp[n][m] = true;
    for (int i = n - 1; i >= 0; --i) {
        dp[i][m] = dp[i + 1][m] && (a[i] == t[i + m]);
    }

    for (int j = m - 1; j >= 0; --j) {
        dp[n][j] = dp[n][j + 1] && (b[j] == t[n + j]);
    }

    for (int i = n - 1; i >= 0; --i) {
        for (int j = m - 1; j >= 0; --j) {
            if (a[i] != b[j]) {
                if (a[i] == t[i + j]) {
                    dp[i][j] = dp[i + 1][j];
                } else if (b[j] == t[i + j]) {
                    dp[i][j] = dp[i][j + 1];
                }
            } else {
                if (a[i] != t[i + j]) continue;

                dp[i][j] = dp[i + 1][j] || dp[i][j + 1];
            }
        }
    }

    return dp[0][0];
}

// 1696. Jump Game VI

int maxResult(std::vector<int> &nums, int K) {
    int n = nums.size();

    std::vector<int> dp(n, 0);

    dp[0] = nums[0];

    std::priority_queue<std::pair<int, int>> pq;
    pq.push(std::pair<int, int>{dp[0], 0});

    for (int i = 1; i < n; ++i) {
        auto maxPrev = pq.top();
        while (maxPrev.second < i - K) {
            pq.pop();

            if (!pq.empty()) {
                maxPrev = pq.top();
            }
        }

        dp[i] = maxPrev.first + nums[i];

        pq.push({dp[i], i});
    }

    return dp[n - 1];
}

// 823. Binary Trees With Factors

int numFactoredBinaryTrees(std::vector<int> &arr) {
    // j > i
    // if (arr[j] % arr[i] == 0)
    // then dp[arr[j]] += dp[arr[i]] * dp[arr[j] / arr[i]]

    std::sort(arr.begin(), arr.end());

    int n = arr.size();

    std::unordered_map<int, long long> dp;

    long long total = 0;
    for (int i = 0; i < n; ++i) {
        dp[arr[i]] = 1;
        for (int j = 0; j < i; ++j) {
            if (arr[i] % arr[j] == 0) {
                dp[arr[i]] = (dp[arr[i]] + dp[arr[j]] * dp[arr[i] / arr[j]]) % MOD;
            }
        }
        total = (total + dp[arr[i]]) % MOD;
    }

    return total;
}

// 2707. Extra Characters in a String

bool startsWith(std::string &s, std::string &t, int pos) {
    if (t.length() - pos < s.length()) return false;
    for (int i = 0; pos + i < t.length() && i < s.length(); ++i) {
        if (s[i] != t[pos + i]) {
            return false;
        }
    }

    return true;
}

int dfs(std::string &s, std::vector<std::string> &dict, int start, std::vector<int> &memo) {
    if (start >= s.length()) {
        return 0;
    }

    if (memo[start] != -1) return memo[start];

    memo[start] = INF;
    int uncovered = 0;
    for (int i = start; i < s.length(); ++i) {
        for (auto word: dict) {

            if (startsWith(word, s, i)) {
                memo[start] = std::min(memo[start], dfs(s, dict, i + word.length(), memo) + uncovered);
            }
        }

        ++uncovered;
    }

    if (memo[start] == INF) {
        return memo[start] = s.length() - start;
    }

    return memo[start];
}

int minExtraChar(std::string s, std::vector<std::string> &dictionary) {
    std::vector<int> memo(s.length(), -1);
    return dfs(s, dictionary, 0, memo);
}

// 357. Count Numbers with Unique Digits

long long choose(int n, int k) {
    if (k > n) return 0;
    if (n == k) return 1;
    if (k <= 0 || n <= 0) return 1;
    return choose(n - 1, k - 1) + choose(n - 1, k);
}

long long fact(int n) {
    if (n <= 1) return 1;
    return n * fact(n - 1);
}

int countNumbersWithUniqueDigits(int n) {
    // fix last digit as 1, then all the rest can be 9!
    // + fix prelast digit as one of 9, all the rest can be 9!
    // + fix preprelast digit as one of 9, all the rest can be 9!

    long long count = 1;

    while (n > 0) {
        count += 9 * choose(9, --n) * fact(n);
    }

    return count;
}

// 638. Shopping Offers

bool canApply(std::vector<int> &offer, std::vector<int> &needs) {
    for (int i = 0; i < needs.size(); ++i) {
        if (needs[i] - offer[i] < 0) return false;
    }
    return true;
}

void apply(std::vector<int> &offer, std::vector<int> &needs, int sign) {
    for (int i = 0; i < needs.size(); ++i) {
        needs[i] += sign * offer[i];
    }
}

int backtrack(std::vector<int> &price, std::vector<std::vector<int>> &special, std::vector<int> &needs,
              std::map<std::vector<int>, int> &memo) {
    // try to use offers while you still can,
    // if you cannot then just add what is necessary

    int n = price.size();

    if (memo.contains(std::vector<int>(needs))) return memo[std::vector<int>(needs)];

    memo[needs] = INF;
    for (auto &offer: special) {
        if (canApply(offer, needs)) {
            int ans = memo[needs];
            apply(offer, needs, -1);
            ans = std::min(ans, backtrack(price, special, needs, memo) + offer[n]);
            apply(offer, needs, 1);
            memo[needs] = ans;
        }
    }

    int diff = 0;
    for (int i = 0; i < n; ++i) {
        if (needs[i] > 0) {
            diff += needs[i] * price[i];
        }
    }

    return memo[needs] = std::min(memo[needs], diff);
}

int shoppingOffers(std::vector<int> &price, std::vector<std::vector<int>> &special, std::vector<int> &needs) {
    // price[i] - price of i-th item
    // special[i][j] - number of j-th item in i-th offer
    // special[i][n] - cost of i-th offer
    // needs[i] - number of i-th item

    std::map<std::vector<int>, int> memo;

    return backtrack(price, special, needs, memo);
}

// 2369. Check if There is a Valid Partition For The Array

bool dfs(std::vector<int> &nums, int i, std::vector<int> &memo) {
    if (i >= nums.size()) return true;

    if (memo[i] != -1) return memo[i];

    if (i + 1 >= nums.size()) return false;

    bool ans = false;
    if (nums[i] == nums[i + 1]) {
        ans = ans || dfs(nums, i + 2, memo);
    }

    if (i + 2 >= nums.size()) return memo[i] = ans;

    if (nums[i] == nums[i + 1] && nums[i + 1] == nums[i + 2] ||
        nums[i + 2] - nums[i + 1] == nums[i + 1] - nums[i] &&
        nums[i + 1] - nums[i] == 1) {
        ans = ans || dfs(nums, i + 3, memo);
    }

    return memo[i] = ans;
}

bool validPartition(std::vector<int> &nums) {
    std::vector<int> memo(nums.size(), -1);
    return dfs(nums, 0, memo);
}

// bottom-up
bool validPartition2(std::vector<int> &nums) {
    int n = nums.size();

    std::vector<bool> dp(4, false);
    dp[0] = true;
    dp[1] = false;
    dp[2] = nums[0] == nums[1];

    for (int i = 2; i < n; ++i) {
        bool two = nums[i] == nums[i - 1];
        bool three = (two && nums[i] == nums[i - 2]) ||
                     (nums[i] - 1 == nums[i - 1] && nums[i] - 2 == nums[i - 2]);

        dp[(i + 1) % 4] = (dp[(i - 1) % 4] && two) || (dp[(i - 2) % 4] && three);
    }

    return dp[n % 4];
}

// 2767. Partition String Into Minimum Beautiful Substrings

int minimumBeautifulSubstrings(std::string s) {
    std::unordered_set < std::string > pows{
            "1",
            "101",
            "11001",
            "1111101",
            "1001110001",
            "110000110101",
            "11110100001001"
    };

    int n = s.length();

    std::vector<int> dp(n + 1, INF);

    dp[0] = 0;

    for (int i = 0; i < n; ++i) {
        if (s[i] == '0') continue;

        std::string curr = "";
        for (int j = i; j < n; ++j) {
            curr += s[j];
            if (pows.contains(curr)) {
                dp[j + 1] = std::min(dp[j + 1], dp[i] + 1);
            }
        }
    }

    return dp[n] >= INF ? -1 : dp[n];
}

// 790. Domino and Tromino Tiling

long long countTilings(int n, int trimino, std::vector<std::vector<int>> &memo) {
    if (n < 0) return 0;

    if (n == 0 && trimino == 0) return 1;

    if (memo[n][trimino] != -1) return memo[n][trimino];

    if (trimino == 1) {
        return memo[n][trimino] = (countTilings(n - 1, 0, memo) + countTilings(n - 1, 2, memo)) % MOD;
    } else if (trimino == 2) {
        return memo[n][trimino] = (countTilings(n - 1, 0, memo) + countTilings(n - 1, 1, memo)) % MOD;
    }

    return memo[n][trimino] =
                   (countTilings(n - 1, 0, memo) +
                    countTilings(n - 2, 0, memo) +
                    countTilings(n - 2, 1, memo) +
                    countTilings(n - 2, 2, memo)) % MOD;
}

int numTilings(int n) {
    std::vector<std::vector<int>> memo(n + 1, std::vector<int>(3, -1));
    return countTilings(n, 0, memo);
}

// 1024. Video Stitching

int dfs(std::vector<std::vector<int>> &clips, int pos, int time, int limit, std::vector<std::vector<int>> &memo) {
    if (pos >= clips.size() || time - limit <= 0) {
        if (time - limit <= 0) return 0;
        return INF;
    }

    if (memo[pos][limit] != -1) return memo[pos][limit];

    memo[pos][limit] = INF;
    for (int i = pos; i < clips.size(); ++i) {
        if (clips[i][0] <= limit) {
            memo[pos][limit] = std::min(memo[pos][limit],
                                        dfs(clips, i + 1, time, std::max(limit, clips[i][1]), memo) + 1);
        }
    }

    return memo[pos][limit];
}

int videoStitching(std::vector<std::vector<int>> &clips, int time) {
    std::sort(clips.begin(), clips.end());

    int n = clips.size();

    std::vector<std::vector<int>> memo(n, std::vector<int>(101, -1));

    int res = dfs(clips, 0, time, 0, memo);

    if (res == INF) return -1;

    return res;
}

// 718. Maximum Length of Repeated Subarray

int findLength(std::vector<int> &a, std::vector<int> &b) {
    int n = a.size();
    int m = b.size();

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));

    int ans = 0;

    for (int i = n - 1; i >= 0; --i) {
        for (int j = m - 1; j >= 0; --j) {
            if (a[i] == b[j]) {
                dp[i][j] = dp[i + 1][j + 1] + 1;
            }

            ans = std::max(ans, dp[i][j]);
        }
    }

    return ans;
}

// 1262. Greatest Sum Divisible by Three

int maxSumDivThree(std::vector<int> &nums) {
    int n = nums.size();

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(3, 0));

    int ans = 0;
    for (int i = 0; i < n; ++i) {
        for (int r = 0; r < 3; ++r) {
            int sum = dp[i][r] + nums[i];
            dp[i + 1][sum % 3] = std::max(dp[i + 1][sum % 3], sum);
            dp[i + 1][r] = std::max(dp[i + 1][r], dp[i][r]);
        }

        ans = std::max(ans, dp[i + 1][0]);
    }

    return ans;
}

// 1626. Best Team With No Conflicts

int dfs(std::vector<std::pair<int, int>> &players, int start, int prev, std::vector<int> &memo) {
    int n = players.size();

    if (start >= n) return 0;

    if (memo[start] != -1) return memo[start];

    memo[start] = 0;
    for (int i = start; i < n; ++i) {
        if (prev != -1 && players[i].first > players[prev].first && players[i].second < players[prev].second) {
            continue;
        }
        memo[start] = std::max(memo[start], dfs(players, i + 1, i, memo) + players[i].second);
    }

    return memo[start];
}

int bestTeamScore(std::vector<int> &scores, std::vector<int> &ages) {
    int n = scores.size();
    std::vector<std::pair<int, int>> players(n);

    for (int i = 0; i < scores.size(); ++i) {
        players[i] = std::pair<int, int>{ages[i], scores[i]};
    }

    std::sort(players.begin(), players.end());

    std::vector<int> memo(n, -1);

    return dfs(players, 0, -1, memo);
}

// 2222. Number of Ways to Select Buildings

// 0 1 0
// 1 0 1
long long numberOfWays(std::string s) {
    // dp[i][l][0] - number of sequence of length l ending with 0
    // dp[i][l][1] - number of sequence of length l ending with 1
    int n = s.length();

    long long ans = 0;

    long long n0 = 0, n1 = 0, n01 = 0, n10 = 0;

    for (int i = 0; i < n; ++i) {
        int zero = s[i] == '0';
        bool one = s[i] == '1';

        ans += n10 * one + n01 * zero;

        n10 += n1 * zero;
        n01 += n0 * one;

        n0 += zero;
        n1 += one;
    }

    return ans;
}

// 1320. Minimum Distance to Type a Word Using Two Fingers

// alphabet size
const int L = 26;

int cost(int a, int b, std::vector<std::pair<int, int>> &dist) {
    auto [x1, y1] = dist[a];
    auto [x2, y2] = dist[b];
    return std::abs(x2 - x1) + std::abs(y2 - y1);
}

int minimumDistance(std::string word) {
    // dp[i][f1][f2]
    // answer - min(dp[n][f1][f2])

    int n = word.size();

    std::vector<std::pair<int, int>> dist(L);

    const std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    int x = 0, y = 0;

    for (char ch: alphabet) {
        dist[ch - 'A'] = std::make_pair(x, y);

        y++;
        if (y > 5) {
            y = 0;
            x++;
        }
    }

    std::vector<std::vector<std::vector<int>>> dp(
            n + 1,
            std::vector<std::vector<int>>(L, std::vector<int>(L, INF))
    );

    for (int f1 = 0; f1 < L; ++f1) {
        for (int f2 = 0; f2 < L; ++f2) {
            dp[0][f1][f2] = 0;
        }
    }

    int ans = INF;
    for (int i = 0; i < n; ++i) {
        for (int f1 = 0; f1 < L; ++f1) {
            for (int f2 = 0; f2 < L; ++f2) {
                int wIdx = word[i] - 'A';

                dp[i + 1][wIdx][f2] = std::min({
                                                       dp[i + 1][wIdx][f2],
                                                       dp[i][f1][f2] + cost(f1, wIdx, dist)
                                               });

                dp[i + 1][f1][wIdx] = std::min({
                                                       dp[i + 1][f1][wIdx],
                                                       dp[i][f1][f2] + cost(f2, wIdx, dist)
                                               });

                if (i + 1 == n) {
                    ans = std::min({ans, dp[i + 1][wIdx][f2], dp[i + 1][f1][wIdx]});
                }
            }
        }
    }

    return ans;
}

// 416. Partition Equal Subset Sum

bool canPartition(std::vector<int>& nums) {
    int n = nums.size();

    int total = 0;
    for (int i = 0; i < n; ++i) {
        total += nums[i];
    }

    if (total % 2 != 0) return false;

    int target = total / 2;

    std::vector<int> dp(target + 1, -1);
    dp[0] = INF;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= target; ++j) {
            if (dp[j] != -1 && dp[j] != i && j + nums[i] <= target) {
                if (dp[j + nums[i]] == -1) {
                    dp[j + nums[i]] = i;
                }
            }
        }
    }

    return dp[target] != -1;
}

int main() {
    auto a = std::vector<int>{1, 1, 1, 1};
    std::cout << canPartition(a) << std::endl;
    return 0;
}