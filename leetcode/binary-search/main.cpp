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
const int MOD = 1e9;

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

// 33. Search in Rotated Sorted Array
int searchRotation(std::vector<int> &nums) {
    int n = nums.size();

    int l = 0;
    int r = n - 1;

    while (l < r) {
        int m = (l + r) / 2;

        if (m + 1 >= n || nums[m] > nums[m + 1]) return m;
        else if (nums[m] > nums[r]) {
            l = m + 1;
        } else {
            r = m;
        }
    }

    return l;
}

int bisect(std::vector<int> &nums, int l, int r, int target) {
    if (l > r) return -1;

    int n = nums.size();

    while (l < r) {
        int m = (l + r) / 2;

        if (nums[m] == target) return m;
        else if (nums[m] < target) {
            l = m + 1;
        } else {
            r = m - 1;
        }
    }

    return nums[l] == target ? l : -1;
}

int search(std::vector<int> &nums, int target) {
    int n = nums.size();

    int pivot = searchRotation(nums);

    int i = bisect(nums, 0, pivot, target);
    if (i != -1) return i;

    return bisect(nums, pivot + 1, n - 1, target);
}

// 34. Find First and Last Position of Element in Sorted Array
int findFirst(std::vector<int> &nums, int target) {
    int n = nums.size();

    int l = 0;
    int r = n - 1;

    while (l < r) {
        int m = (l + r) / 2;

        if (nums[m] == target && (m - 1 < 0 || nums[m - 1] != target)) return m;
        if (nums[m] < target) {
            l = m + 1;
        } else {
            r = m - 1;
        }
    }

    return nums[l] == target ? l : -1;
}

int findLast(std::vector<int> &nums, int target) {
    int n = nums.size();

    int l = 0;
    int r = n - 1;

    while (l < r) {
        int m = (l + r) / 2;

        if (nums[m] == target && (m + 1 >= n || nums[m + 1] != target)) return m;
        if (nums[m] > target) {
            r = m - 1;
        } else {
            l = m + 1;
        }
    }

    return nums[l] == target ? l : -1;
}

std::vector<int> searchRange(std::vector<int> &nums, int target) {
    if (nums.size() == 0) return std::vector<int>{-1, -1};

    return std::vector<int>{findFirst(nums, target), findLast(nums, target)};
}

// 153. Find Minimum in Rotated Sorted Array

int findMin(std::vector<int> &nums) {
    if (nums.size() < 2) return nums[0];
    int pivot = searchRotation(nums);
    if (nums[pivot] < nums[pivot + 1]) return nums[pivot];
    return nums[pivot + 1];
}

// 4. Median of Two Sorted Arrays

double findMedianSortedArrays(std::vector<int> &a, std::vector<int> &b) {
    int n = a.size();
    int m = b.size();

    int total = n + m;

    int half = total / 2;

    if (n < m) {
        std::swap(a, b);
        std::swap(n, m);
    }

    int l = 0;
    int r = m - 1;

    while (true) {
        int i = (l + r) / 2;

        if (l + r < 0) {
            i = -1;
        }

        // index of end element in A
        int j = half - (i + 1) - 1;

        int a_left = j >= 0 ? a[j] : -INF;
        int a_right = j + 1 < n ? a[j + 1] : INF;

        int b_left = i >= 0 ? b[i] : -INF;
        int b_right = i + 1 < m ? b[i + 1] : INF;

        if (a_left <= b_right && b_left <= a_right) {
            if (total % 2 != 0) {
                return std::min(a_right, b_right);
            } else {
                return (double) (std::max(a_left, b_left) + std::min(a_right, b_right)) / 2.0;
            }
        } else if (a_left > b_right) {
            l = i + 1;
        } else {
            r = i - 1;
        }
    }
}

// 1712. Ways to Split Array Into Three Subarrays

int findPivot(std::vector<int> &prefix, int n, int border, bool leftmost) {
    int l = 0;
    int r = border - 1;

    int rightSum = prefix[n - 1] - prefix[border];

    int res = -1;

    while (l <= r) {
        int m = (l + r) / 2;

        int leftSum = prefix[m];
        int midSum = prefix[n - 1] - rightSum - leftSum;
        if (midSum <= rightSum && midSum >= leftSum) {
            res = m;
            if (leftmost) {
                r = m - 1;
            } else {
                l = m + 1;
            }
        } else if (midSum <= rightSum) {
            r = m - 1;
        } else {
            l = m + 1;
        }
    }

    return res;
}

int waysToSplit(std::vector<int> &nums) {
    int n = nums.size();

    std::vector<int> prefix(n, 0);
    prefix[0] = nums[0];
    for (int i = 1; i < n; ++i) {
        prefix[i] = prefix[i - 1] + nums[i];
    }

    int res = 0;
    for (int r = n - 2; r >= 0; --r) {
        int leftPivot = findPivot(prefix, n, r, true);
        int rightPivot = findPivot(prefix, n, r, false);

        if (leftPivot == -1 || rightPivot == -1) continue;

        res = (res + (rightPivot - leftPivot + 1) % MOD) % MOD;
    }

    return res;
}

// 275. H-Index II

int hIndex(std::vector<int>& citations) {
    int n = citations.size();

    int l = 0;
    int r = citations.size() - 1;

    while (l <= r) {
        int m = (l + r) / 2;

        if (citations[m] == n - m) {
            return citations[m];
        } else if (citations[m] > n - m) {
            r = m - 1;
        } else {
            l = m + 1;
        }
    }

    return n - (r + 1);
}

int main() {
    auto v = std::vector<int>{1, 2, 2, 2, 5, 0};
    std::cout << waysToSplit(v) << std::endl;
    return 0;
}