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

// 766. Toeplitz Matrix

bool isToeplitzMatrix(std::vector<std::vector<int>> &matrix) {
    int n = matrix.size();
    int m = matrix[0].size();

    int diags = n + m - 1;

    for (int k = 0; k < m; ++k) {
        for (int i = 0; i < n; ++i) {
            int j = i + k;
            if (j >= m) break;

            if (i - 1 >= 0 && j - 1 >= 0) {
                if (matrix[i - 1][j - 1] != matrix[i][j]) return false;
            }
        }
    }

    for (int k = 1; k < n; ++k) {
        for (int j = 0; j < m; ++j) {
            int i = j + k;
            if (i >= n) break;

            if (i - 1 >= 0 && j - 1 >= 0) {
                if (matrix[i - 1][j - 1] != matrix[i][j]) return false;
            }
        }
    }

    return true;
}

// 299. Bulls and Cows

std::string getHint(std::string secret, std::string guess) {
    int bulls = 0;
    for (int i = 0; i < secret.length(); ++i) {
        if (secret[i] == guess[i]) ++bulls;
    }

    std::vector<int> dict(10, 0);
    for (int i = 0; i < secret.length(); ++i) {
        if (secret[i] == guess[i]) continue;
        ++dict[secret[i] - '0'];
    }

    int cows = 0;
    for (int i = 0; i < secret.length(); ++i) {
        if (dict[guess[i] - '0'] != 0 && secret[i] != guess[i]) {
            ++cows;
            --dict[guess[i] - '0'];
        }
    }

    return std::to_string(bulls) + "A" + std::to_string(cows) + "B";
}

// 980. Unique Paths III

bool bitSet(int n, int i) {
    return n & (1 << i);
}

int setBit(int n, int i) {
    return n | (1 << i);
}

bool isValid(int i, int j, std::vector<std::vector<int>> &grid, int visited) {
    int n = grid.size();
    int m = grid[0].size();

    if (i < 0 || j < 0 || i >= n || j >= m) {
        return false;
    }

    return grid[i][j] != -1 && !bitSet(visited, i * m + j);
}

int dfs(int i, int j, std::vector<std::vector<int>> &grid, int visited, int target) {
    if (grid[i][j] == 2 && target == 0) return 1;

    int n = grid.size();
    int m = grid[0].size();

    std::vector<std::pair<int, int>> dirs{{-1, 0},
                                          {0,  -1},
                                          {0,  1},
                                          {1,  0}};

    int count = 0;
    for (auto [di, dj]: dirs) {
        if (!isValid(i + di, j + dj, grid, visited)) continue;

        count += dfs(i + di, j + dj, grid, setBit(visited, (i + di) * m + (j + dj)), target - 1);
    }

    return count;
}

int uniquePathsIII(std::vector<std::vector<int>> &grid) {
    int n = grid.size();
    int m = grid[0].size();

    int start_i = -1;
    int start_j = -1;

    int target = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (grid[i][j] == 0) ++target;
            if (grid[i][j] != 1) continue;
            start_i = i;
            start_j = j;
        }
    }

    return dfs(start_i, start_j, grid, setBit(0, start_i * m + start_j), target + 1);
}

// 807. Max Increase to Keep City Skyline

int maxIncreaseKeepingSkyline(std::vector<std::vector<int>> &grid) {
    int n = grid.size();

    std::vector<int> rows(n, 0);
    std::vector<int> cols(n, 0);
    for (int i = 0; i < n; ++i) {
        int row_max = 0;
        for (int j = 0; j < n; ++j) {
            cols[j] = std::max(cols[j], grid[i][j]);
            rows[i] = std::max(rows[i], grid[i][j]);
        }
    }

    int res = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            res += std::min(rows[i], cols[j]) - grid[i][j];
        }
    }
    return res;
}

// 73. Set Matrix Zeroes

void setZeroes(std::vector<std::vector<int>> &matrix) {
    int n = matrix.size();
    int m = matrix[0].size();

    bool col0 = true;

    for (int i = 0; i < n; ++i) {
        if (matrix[i][0] == 0) {
            col0 = false;
        }
        for (int j = 1; j < m; ++j) {
            if (matrix[i][j] == 0) {
                matrix[i][0] = 0;
                matrix[0][j] = 0;
            }
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        for (int j = m - 1; j >= 1; --j) {
            if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                matrix[i][j] = 0;
            }
        }
        if (!col0) matrix[i][0] = 0;
    }

    return;
}

// 419. Battleships in a Board

int countBattleships(std::vector<std::vector<char>> &board) {
    int n = board.size();
    int m = board[0].size();

    int count = 0;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (board[i][j] != 'X') continue;

            if ((i - 1 >= 0 && board[i - 1][j] == 'X') || (j - 1 >= 0 && board[i][j - 1] == 'X')) continue;

            ++count;
        }
    }

    return count;
}

// 1016. Binary String With Substrings Representing 1 To N

std::string toBin(int n) {
    std::string res;

    while (n > 0) {
        res += (n % 2) + '0';
        n = n >> 1;
    }

    std::reverse(res.begin(), res.end());

    return res;
}

bool queryString(std::string s, int N) {
    if (N > 2047) return false;

    for (int n = 1; n <= N; ++n) {
        std::string bin = toBin(n);

        bool fail = true;
        for (int start = 0; start < s.length(); ++start) {
            bool found = true;
            for (int i = 0; i < bin.length(); ++i) {
                if (s[start + i] != bin[i]) {
                    found = false;
                    break;
                }
            }

            if (found) {
                fail = false;
                break;
            }
        }

        if (fail) return false;
    }

    return true;
}

// 135. Candy

int candy(std::vector<int> &ratings) {
    int n = ratings.size();

    if (n == 1) return 1;

    std::vector<int> candies(n, 1);

    int candy = n;
    int giving = 0;
    for (int i = 1; i < n; ++i) {
        if (ratings[i] > ratings[i - 1]) {
            ++giving;
        } else {
            giving = 0;
        }

        candy += giving;
        candies[i] += giving;
    }

    for (int i = n - 2; i >= 0; --i) {
        if (ratings[i] > ratings[i + 1] && candies[i] <= candies[i + 1]) {
            candy += (candies[i + 1] - candies[i] + 1);
            candies[i] = candies[i + 1] + 1;
        }
    }

    return candy;
}

// 93. Restore IP Addresses

std::string toIP(std::vector<std::string> &domains) {
    int last = domains.size() - 1;

    std::string res = "";
    for (int i = 0; i < last; ++i) {
        res += domains[i] + ".";
    }

    return res + domains[last];
}

void dfs(int start, std::string &s, std::vector<std::string> &domains, std::unordered_set<std::string> &res) {
    if ((domains.size() >= 4 && start < s.length())) return;

    if (start >= s.length()) {
        if (domains.size() < 4) return;
        res.insert(toIP(domains));
        return;
    }

    std::string curr = "";
    for (int i = start; i < start + 3 && i < s.length(); ++i) {
        curr += s[i];

        if (std::stoi(curr) > 255) break;

        domains.emplace_back(curr);

        dfs(i + 1, s, domains, res);

        domains.pop_back();

        if (curr == "0") return;
    }
}

std::vector<std::string> restoreIpAddresses(std::string s) {
    if (s.length() > 12) {
        return std::vector<std::string>{};
    }

    std::vector<std::string> domains;

    std::unordered_set < std::string > container;
    dfs(0, s, domains, container);

    std::vector<std::string> res(container.begin(), container.end());

    std::sort(res.begin(), res.end());

    return res;
}

// 1578. Minimum Time to Make Rope Colorful

int minCost(std::string colors, std::vector<int> &neededTime) {
    auto prev_color = colors[0];

    int min_time = 0;
    int prev_sum = neededTime[0];
    int max_val = neededTime[0];
    for (int i = 1; i < colors.size(); ++i) {
        if (prev_color != colors[i]) {
            min_time += prev_sum - max_val;
            prev_sum = 0;
            max_val = 0;
        }
        prev_color = colors[i];
        prev_sum += neededTime[i];
        max_val = std::max(max_val, neededTime[i]);
    }

    return min_time + prev_sum - max_val;
}

// 50. Pow(x, n)

double myPow(double x, long long n) {
    if (n == 0) return 1;

    if (n < 0) {
        return 1.0 / myPow(x, -n);
    }

    if (n % 2 == 0) {
        double p = myPow(x, n / 2);
        return p * p;
    } else {
        return x * myPow(x, n - 1);
    }
}

// 915. Partition Array into Disjoint Intervals

int partitionDisjoint(std::vector<int>& nums) {
    // keep a vector for the left part
    // keep min element for the right part
    // whenever max from left is less than min from right -> calculate minLen
    int n = nums.size();

    std::vector<int> maxLeft(n, 0);
    maxLeft[0] = nums[0];
    for (int i = 1; i < n; ++i) {
        maxLeft[i] = std::max(maxLeft[i - 1], nums[i]);
    }

    int minRight = INF;
    int minLen = n;
    for (int i = n - 1; i >= 0; --i) {
        if (maxLeft[i] <= minRight) {
            minLen = std::min(minLen, i + 1);
        }

        minRight = std::min(minRight, nums[i]);
    }

    return minLen;
}

// 554. Brick Wall

int leastBricks(std::vector<std::vector<int>>& wall) {
    std::unordered_map<int, int> pref;

    int n = wall.size();

    for (int i = 0; i < n; ++i) {
        int prefSum = 0;
        for (int j = 0; j < wall[i].size() - 1; ++j) {
            prefSum += wall[i][j];
            ++pref[prefSum];
        }
    }

    int count = 0;
    for (auto [w, c] : pref) {
        count = std::max(count, c);
    }

    return n - count;
}

int main() {
    std::vector<int> v{1, 2, 2};
    return 0;
}