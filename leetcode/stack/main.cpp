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

// 84. Largest Rectangle in Histogram

int largestRectangleArea(std::vector<int>& heights) {
    heights.emplace_back(0);

    std::stack<int> hist;
    std::stack<int> indexes;

    int maxArea = 0;
    for (int i = 0; i < heights.size(); ++i) {
        int prev = i;
        while (!hist.empty() && heights[i] < hist.top()) {
            int h = hist.top();
            hist.pop();

            prev = indexes.top();
            indexes.pop();

            maxArea = std::max(maxArea, h * (i - prev));
        }

        hist.push(heights[i]);
        indexes.push(prev);
    }

    return maxArea;
}

// 1504. Count Submatrices With All Ones

int numSubmat(std::vector<std::vector<int>> &mat) {
    int n = mat.size();
    int m = mat[0].size();

    int res = 0;

    auto h = std::vector<int>(m, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            h[j] = (mat[i][j] == 0 ? 0 : h[j] + 1);
        }

        auto sum = std::vector<int>(m, 0);
        std::stack<int> bars;

        for (int j = 0; j < m; ++j) {
            while (!bars.empty() && h[bars.top()] >= h[j]) {
                bars.pop();
            }

            if (!bars.empty()) {
                int prevIndex = bars.top();
                sum[j] = sum[prevIndex];
                sum[j] += (j - prevIndex) * h[j];
            } else {
                sum[j] = h[j] * (j + 1);
            }

            bars.push(j);
        }

        for (int j = 0; j < m; ++j) {
            res += sum[j];
        }
    }

    return res;
}


int main() {
    auto v = std::vector<int>{2, 1, 5, 6, 2, 3};
    std::cout << largestRectangleArea(v) << std::endl;
    return 0;
}