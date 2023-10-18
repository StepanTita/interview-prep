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

int largestRectangleArea(std::vector<int> &heights) {
    std::stack<int> bars;
    std::stack<int> indexes;

    int ans = 0;
    for (int i = 0; i < heights.size(); ++i) {
        if (bars.empty() || heights[i] >= bars.top()) {
            bars.push(heights[i]);
            indexes.push(i);
        } else {
            int nextIndex = i;
            while (!bars.empty() && bars.top() > heights[i]) {
                ans = std::max(ans, (i - indexes.top()) * bars.top());

                nextIndex = indexes.top();
                bars.pop();
                indexes.pop();
            }
            bars.push(heights[i]);
            indexes.push(nextIndex);
        }
    }

    int lastIndex = heights.size();

    while (!bars.empty()) {
        auto currIndex = indexes.top();
        ans = std::max(ans, (lastIndex - currIndex) * bars.top());
        bars.pop();
        indexes.pop();
    }

    return ans;
}


int main() {
    auto v = std::vector<int>{2, 1, 5, 6, 2, 3};
    std::cout << largestRectangleArea(v) << std::endl;
    return 0;
}