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

// 2900. Longest Unequal Adjacent Groups Subsequence I

std::vector<std::string>
getWordsInLongestSubsequence(int n, std::vector<std::string> &words, std::vector<int> &groups) {
    // so I need to select the longest subsequence from words, such that
    // in groups the next element of this subsequence is different
    std::vector<std::string> odd;

    int prev = groups[0];

    odd.emplace_back(words[0]);
    for (int i = 1; i < n; ++i) {
        if (prev != groups[i]) {
            odd.emplace_back(words[i]);
            prev = groups[i];
        }
    }

    return odd;
}

// 1090. Largest Values From Labels

int largestValsFromLabels(std::vector<int> &values, std::vector<int> &labels, int numWanted, int useLimit) {
    int n = values.size();

    std::vector<std::pair<int, int>> pairs;
    for (int i = 0; i < n; ++i) {
        pairs.emplace_back(std::pair<int, int>{values[i], labels[i]});
    }

    std::sort(pairs.begin(), pairs.end());

    std::unordered_map<int, int> used;

    int score = 0;
    for (int i = n - 1; i >= 0 && numWanted > 0; --i) {
        if (used[pairs[i].second] < useLimit) {
            score += pairs[i].first;
            ++used[pairs[i].second];
            --numWanted;
        }
    }

    return score;
}

// 1402. Reducing Dishes

int maxSatisfaction(std::vector<int>& sat) {
    std::sort(sat.begin(), sat.end(), std::greater<>());

    int n = sat.size();

    int res = 0;
    int curr = 0;
    for (int i = 0; i < n && sat[i] > -curr; ++i) {
        curr += sat[i];
        res += curr;
    }

    return std::max(0, res);
}

// 31. Next Permutation

void nextPermutation(std::vector<int>& nums) {
    int n = nums.size();

    int pivot = -1;
    for (int i = n - 2; i >= 0; --i) {
        if (nums[i] < nums[i + 1]) {
            pivot = i;
            break;
        }
    }

    if (pivot == -1) {
        std::reverse(nums.begin(), nums.end());
        return;
    }

    for (int i = n - 1; i > pivot; --i) {
        if (nums[i] > nums[pivot]) {
            std::swap(nums[i], nums[pivot]);
            break;
        }
    }

    std::reverse(nums.begin() + pivot + 1, nums.end());
}

// 1053. Previous Permutation With One Swap

std::vector<int> prevPermOpt1(std::vector<int>& arr) {
    int n = arr.size();

    int pivot = n - 2;
    for (; pivot >= 0; --pivot) {
        if (arr[pivot] > arr[pivot + 1]) {
            break;
        }
    }

    if (pivot < 0) return arr;

    int currMaxIdx = pivot + 1;
    for (int i = pivot + 1; i < n; ++i) {
        if (arr[i] > arr[currMaxIdx] && arr[i] < arr[pivot]) {
            currMaxIdx = i;
        }
    }

    std::swap(arr[pivot], arr[currMaxIdx]);
    return arr;
}

std::vector<std::string> fullJustify(std::vector<std::string>& words, int maxWidth) {
    int n = words.size();

    std::vector<std::string> res;

    std::vector<std::string> line;
    for (int i = 0; i < n;) {
        line.emplace_back(words[i]);
        int textLen = words[i].length();
        int totalSpLen = 0;
        ++i;

        int nextWordLen = 0;
        if (i < n) {
            nextWordLen = words[i].length();
        }
        while (i < n && textLen + nextWordLen + totalSpLen + 1 <= maxWidth) {
            textLen += nextWordLen;
            ++totalSpLen;
            line.emplace_back(words[i]);
            ++i;
            if (i < n) {
                nextWordLen = words[i].length();
            }
        }

        int lineLen = line.size();

        int extraSpaces = 0;

        int spLen = 1;
        if (lineLen > 1) {
            extraSpaces = (maxWidth - textLen - totalSpLen) % (lineLen - 1);
            int distSpaces = (maxWidth - textLen - totalSpLen) / (lineLen - 1);
            spLen = distSpaces + 1;
        }

        std::string space = "";
        while (spLen > 0) {
            space += " ";
            --spLen;
        }

        std::string lineStr = "";

        if (i >= n) {
            space = " ";
        }
        for (int l = 0; l < lineLen; ++l) {
            lineStr += line[l];
            if (l + 1 < lineLen) {
                lineStr += space;
                if (i < n && extraSpaces-- > 0) {
                    lineStr += " ";
                }
            }
        }
        while (lineStr.length() < maxWidth) {
            lineStr += " ";
        }

        res.emplace_back(lineStr);

        line.clear();
    }

    return res;
}

// 2645. Minimum Additions to Make Valid String

int addMinimum(std::string word) {
    std::string abc = "abc";
    int count = 0;

    int j = 0;
    for (int i = 0; i < word.size(); j = (j + 1) % 3) {
        if (word[i] != abc[j]) {
            ++count;
        } else {
            ++i;
        }
    }

    if (j != 0)
        count += 3 - j;

    return count;
}

// 2522. Partition String Into Substrings With Values at Most K

int minimumPartition(std::string s, int k) {
    int n = s.length();

    int res = 1;

    long long val = 0;
    for (int i = 0; i < n; ++i) {
        val = 10 * val + (s[i] - '0');

        if (val > k) {
            ++res;
            val = s[i] - '0';
        }

        if (val > k) return -1;
    }

    return res;
}

// 2086. Minimum Number of Food Buckets to Feed the Hamsters

int minimumBuckets(std::string h) {
    int n = h.length();

    int count = 0;
    for (int i = 0; i < n; ++i) {
        if (h[i] == 'H') {
            if (i > 0 && h[i - 1] == 'D') continue;

            if (i + 1 < n && h[i + 1] == '.') {
                h[i + 1] = 'D';
                ++count;
            } else if (i - 1 >= 0 && h[i - 1] == '.') {
                h[i - 1] = 'D';
                ++count;
            } else return -1;
        }
    }

    return count;
}

int main() {
    addMinimum("b");
    return 0;
}