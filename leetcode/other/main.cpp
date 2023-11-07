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