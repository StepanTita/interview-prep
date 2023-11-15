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

int main() {
    queryString("10010111100001110010", 10);
    return 0;
}