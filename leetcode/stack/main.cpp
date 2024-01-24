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

// 1249. Minimum Remove to Make Valid Parentheses

std::string minRemoveToMakeValid(std::string s) {
    std::stack<char> st;

    int balance = 0;

    for (int i = 0; i < s.length(); ++i) {
        if (s[i] == ')' && balance - 1 < 0) continue;

        balance += s[i] == '(';
        balance -= s[i] == ')';
        st.push(s[i]);
    }

    std::string res = "";

    balance = 0;
    while (!st.empty()) {
        auto curr = st.top();
        st.pop();

        if (curr == '(' && balance - 1 < 0) continue;
        res += curr;

        balance -= curr == '(';
        balance += curr == ')';
    }

    std::reverse(res.begin(), res.end());

    return res;
}

int setBit(int n, int i) {
    return n | (1 << i);
}

int unsetBit(int n, int i) {
    return n & (~(1 << i));
}

int bitSet(int n, int i) {
    return (n & (1 << i)) != 0;
}

int toNum(char c) {
    return c - 'a';
}

// 316. Remove Duplicate Letters

std::string removeDuplicateLetters(std::string s) {
    int n = s.length();

    int used = 0;

    std::vector<int> lastIdx(26, 0);
    for (int i = 0; i < n; ++i) {
        lastIdx[toNum(s[i])] = i;
    }

    std::stack<char> st;
    for (int i = 0; i < n; ++i) {
        if (bitSet(used, toNum(s[i])))
            continue;

        while (!st.empty() && st.top() > s[i] && lastIdx[toNum(st.top())] > i) {
            used = unsetBit(used, toNum(st.top()));
            st.pop();
        }

        st.push(s[i]);
        used = setBit(used, toNum(s[i]));
    }

    std::string res;
    while (!st.empty()) {
        res += st.top();
        st.pop();
    }

    std::reverse(res.begin(), res.end());

    return res;
}

// 1019. Next Greater Node In Linked List

struct ListNode {
    int val;
    ListNode *next;

    ListNode() : val(0), next(nullptr) {}

    ListNode(int x) : val(x), next(nullptr) {}

    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

std::vector<int> nextLargerNodes(ListNode *head) {
    std::vector<int> res;

    auto curr = head;

    // decreasing stack
    std::stack<std::pair<int, int>> st;
    int idx = 0;
    res.emplace_back(0);

    while (curr != NULL) {
        if (st.empty() || st.top().first >= curr->val) {
            st.push({curr->val, idx});
        } else {
            while (!st.empty() && st.top().first < curr->val) {
                res[st.top().second] = curr->val;
                st.pop();
            }

            st.push({curr->val, idx});
        }

        curr = curr->next;
        ++idx;
        if (curr != NULL)
            res.emplace_back(0);
    }

    return res;
}

int main() {
    auto v = std::vector<int>{2, 1, 5, 6, 2, 3};
    std::cout << largestRectangleArea(v) << std::endl;
    return 0;
}