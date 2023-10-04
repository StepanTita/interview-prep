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

int main() {
    auto a = std::vector<int>{2, 5, 1, 2, 5};
    auto b = std::vector<int>{10, 5, 2, 1, 5, 2};
    auto v = longestPalindromeSubseq("bbbab");
    std::cout << v << std::endl;
    return 0;
}