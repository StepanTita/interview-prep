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

// 35. Search Insert Position
int searchInsert(std::vector<int> &nums, int target) {
    int l = 0;
    int r = nums.size() - 1;
    while (l < r) {
        int m = (l + r) / 2;
        if (nums[m] == target) {
            return m;
        } else if (nums[m] < target) {
            l = m + 1;
        } else {
            r = m - 1;
        }
    }
    if (target > nums[l]) {
        return l + 1;
    }
    return l;
}

// 74. Search a 2D Matrix
bool bisect(std::vector<int> &a, int target) {
    int l = 0;
    int r = a.size() - 1;

    while (l < r) {
        int m = (l + r) / 2;

        if (a[m] == target) return true;
        else if (a[m] < target) {
            l = m + 1;
        } else {
            r = m - 1;
        }
    }

    return a[l] == target;
}

bool searchMatrix(std::vector<std::vector<int>> &matrix, int target) {
    int l = 0;
    int r = matrix.size() - 1;

    int n = matrix[0].size();

    while (l < r) {
        int m = (l + r) / 2;

        if (matrix[m][0] <= target && matrix[m][n - 1] >= target) {
            return bisect(matrix[m], target);
        } else if (matrix[m][n - 1] < target) {
            l = m + 1;
        } else {
            r = m - 1;
        }
    }

    return bisect(matrix[l], target);
}

// 240. Search a 2D Matrix II
bool searchMatrix2D(std::vector<std::vector<int>> &matrix, int target) {
    int n = matrix.size();
    int m = matrix[0].size();

    int row = 0;
    int col = m - 1;

    while (row < n && col >= 0) {
        if (matrix[row][col] == target) return true;
        else if (matrix[row][col] < target) {
            ++row;
        } else if (matrix[row][col] > target) {
            --col;
        }
    }

    return false;
}

// 162. Find Peak Element
int findPeakElement(std::vector<int> &nums) {
    int l = 0;
    int r = nums.size() - 1;

    int n = nums.size();

    while (l < r) {
        int m = (l + r) / 2;

        if ((m + 1 >= n || nums[m] > nums[m + 1]) && (m - 1 < 0 || nums[m - 1] < nums[m])) {
            return m;
        } else if (m + 1 < n && nums[m + 1] > nums[m]) {
            l = m + 1;
        } else if (m - 1 >= 0 && nums[m - 1] > nums[m]) {
            r = m - 1;
        }
    }

    return l;
}