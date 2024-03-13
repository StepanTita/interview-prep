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

int partitionDisjoint(std::vector<int> &nums) {
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

int leastBricks(std::vector<std::vector<int>> &wall) {
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
    for (auto [w, c]: pref) {
        count = std::max(count, c);
    }

    return n - count;
}

// 565. Array Nesting

int arrayNesting(std::vector<int> &nums) {
    int n = nums.size();

    int maxLen = 0;

    std::vector<bool> visited(n, false);
    for (int i = 0; i < n; ++i) {
        int k = i;

        int len = 0;
        while (!visited[k]) {
            visited[k] = true;
            k = nums[k];
            ++len;
            maxLen = std::max(maxLen, len);
        }
    }

    return maxLen;
}

// 2186. Minimum Number of Steps to Make Two Strings Anagram II

int minSteps(std::string s, std::string t) {
    std::vector<int> fs('z' - 'a' + 1, 0);
    for (char c: s) {
        ++fs[c - 'a'];
    }

    for (char c: t) {
        --fs[c - 'a'];
    }

    int ans = 0;
    for (int c = 0; c <= 'z' - 'a'; ++c) {
        ans += std::abs(fs[c]);
    }

    return ans;
}

// 2423. Remove Letter To Equalize Frequency

bool equalFrequency(std::string word) {
    std::unordered_map<int, int> freq;
    for (char c: word) {
        ++freq[c];
    }

    std::unordered_map<int, int> countFreq;

    int countMax = 0;
    int maxFreq = 0;
    int minFreq = INF;
    for (auto [c, f]: freq) {
        if (maxFreq == f) {
            ++countMax;
        }
        if (f > maxFreq) {
            countMax = 1;
        }

        ++countFreq[f];
        maxFreq = std::max(maxFreq, f);
        minFreq = std::min(minFreq, f);
    }

    if (countFreq.size() != 2) {
        if (countFreq.size() == 1) return countFreq.contains(1) || freq.size() == 1;
        return false;
    };

    if (countFreq[1] == 1) return true;

    return maxFreq - minFreq == 1 && countMax == 1;
}

// 997. Find the Town Judge

int findJudge(int n, std::vector<std::vector<int>> &trust) {
    std::vector<int> trustsTo(n, 0);
    std::vector<int> trustsFrom(n, 0);

    for (auto tr: trust) {
        auto to = tr[0];
        auto from = tr[1];

        ++trustsTo[to - 1];
        ++trustsFrom[from - 1];
    }

    for (int i = 0; i < n; ++i) {
        if (trustsTo[i] == 0 && trustsFrom[i] == n - 1) return i + 1;
    }

    return -1;
}

// 435. Non-overlapping Intervals

bool intersect(std::vector<int> &a, std::vector<int> &b) {
    return a[1] > b[0];
}

int eraseOverlapIntervals(std::vector<std::vector<int>> &intervals) {
    std::sort(intervals.begin(), intervals.end());

    int n = intervals.size();
    auto prev = intervals[0];
    int count = 0;
    for (int i = 1; i < n; ++i) {
        auto curr = intervals[i];
        if (intersect(prev, curr)) {
            if (prev[1] > curr[1]) {
                prev = curr;
            }
            ++count;
        } else {
            prev = curr;
        }
    }

    return count;
}

// 792. Number of Matching Subsequences

bool isSubsequence(const std::string &s, std::string &t) {
    int j = 0;
    for (int i = 0; i < t.length(); ++i) {
        if (s[j] == t[i]) {
            ++j;
        }
    }

    return j == s.length();
}

int numMatchingSubseq(std::string s, std::vector<std::string> &words) {
    std::unordered_map<std::string, int> wd;

    for (auto w: words) {
        ++wd[w];
    }

    int count = 0;
    for (auto [w, f]: wd) {
        count += f * isSubsequence(w, s);
    }

    return count;
}

// 1605. Find Valid Matrix Given Row and Column Sums

std::vector<std::vector<int>> restoreMatrix(std::vector<int> &rowSum, std::vector<int> &colSum) {
    int n = rowSum.size();
    int m = colSum.size();

    std::vector<std::vector<int>> mat(n, std::vector<int>(m, 0));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            mat[i][j] = std::min(rowSum[i], colSum[j]);
            rowSum[i] -= mat[i][j];
            colSum[j] -= mat[i][j];
        }
    }

    return mat;
}

// 438. Find All Anagrams in a String

std::vector<int> findAnagrams(std::string s, std::string p) {
    if (s.length() < p.length()) return std::vector<int>{};

    std::unordered_map<char, int> freqP;
    std::unordered_map<char, int> freqS;

    int w = p.length();
    for (int i = 0; i < w; ++i) {
        ++freqP[p[i]];
        --freqP[s[i]];
    }


    std::vector<int> res;

    for (int i = 0; i <= s.length() - w; ++i) {
        bool fail = false;
        for (auto [k, f]: freqP) {
            if (f != 0) {
                fail = true;
                break;
            }
        }
        if (!fail) {
            res.emplace_back(i);
        }

        ++freqP[s[i]];
        --freqP[s[i + w]];
    }

    return res;
}

// 809. Expressive Words

std::pair<std::unordered_map<int, char>, std::unordered_map<int, int>> transform(std::string &s) {
    std::unordered_map<int, char> pos;
    std::unordered_map<int, int> freq;

    int p = 0;
    ++freq[p];
    pos[0] = s[0];
    for (int i = 1; i < s.length(); ++i) {
        if (s[i - 1] != s[i]) {
            pos[++p] = s[i];
        }
        ++freq[p];
    }

    return std::make_pair(pos, freq);
}

bool stretchy(
        std::unordered_map < int, char > &originalPos,
        std::unordered_map < int, int > &originalFreq,

        std::unordered_map < int, char > &pos,
        std::unordered_map < int, int > &freq
) {
    if (freq.size() != originalFreq.size()) return false;

    for (auto [p, c]: originalPos) {
        if (pos[p] != c) {
            return false;
        }

        if (originalFreq[p] < freq[p] || originalFreq[p] < 3) return false;
    }

    return true;
}

int expressiveWords(std::string s, std::vector<std::string> &words) {
    auto [originalPos, originalFreq] = transform(s);

    int count = 0;
    for (int i = 0; i < words.size(); ++i) {
        if (words[i].length() > s.length()) continue;
        auto [pos, freq] = transform(words[i]);
        if (stretchy(originalPos, originalFreq, pos, freq)) {
            ++count;
        }
    }

    return count;
}

// 2 pointers

bool stretchy(
        std::string &s,
        std::string &w
) {
    int n = s.length();
    int m = w.length();

    if (n < m) return false;

    int wi = 0;

    int count = 1;
    for (int i = 0; i < n; ++i) {
        if (wi >= m || s[i] != w[wi]) return false;

        if (i + 1 < n && s[i + 1] == s[i]) {
            ++count;
            continue;
        }

        bool enough = count >= 3;

        while (wi < m && w[wi] == s[i]) {
            --count;
            ++wi;
        }

        if (count < 0 || (count != 0 && !enough)) {
            return false;
        }

        count = 1;
    }

    return wi == w.length();
}

int expressiveWords2(std::string s, std::vector<std::string> &words) {
    int count = 0;
    for (int i = 0; i < words.size(); ++i) {
        if (stretchy(s, words[i])) {
            ++count;
        }
    }

    return count;
}

// 28. Find the Index of the First Occurrence in a String

// KMP (Knuth-Morris-Pratt)

int strStr(std::string haystack, std::string needle) {
    int n = haystack.size();
    int m = needle.size();

    std::vector<int> border(m, 0);

    int prev = 0;
    int curr = 1;
    while (curr < m) {
        if (needle[prev] == needle[curr]) {
            border[curr] = prev + 1;
            ++prev;
            ++curr;
        } else if (prev == 0) {
            border[curr] = 0;
            ++curr;
        } else {
            prev = border[prev - 1];
        }
    }

    int i = 0, j = 0;
    while (i < n) {
        if (haystack[i] == needle[j]) {
            ++i;
            ++j;
        } else {
            if (j == 0) {
                ++i;
            } else {
                j = border[j - 1];
            }
        }

        if (j == m) {
            return i - m;
        }
    }

    return -1;
}

// 218. The Skyline Problem

std::vector<std::vector<int>> getSkyline(std::vector<std::vector<int>> &buildings) {
    std::vector<std::pair<int, int>> points;

    for (auto b: buildings) {
        points.emplace_back(std::make_pair(b[0], -b[2]));
        points.emplace_back(std::make_pair(b[1], b[2]));
    }

    std::sort(points.begin(), points.end());

    std::vector<std::vector<int>> ans;

    std::multiset<int> fallback{0};

    int currH = 0;
    for (int i = 0; i < points.size(); ++i) {
        auto [p, h] = points[i];

        if (h < 0) {
            fallback.insert(-h);
        } else {
            fallback.erase(fallback.find(h));
        }

        auto fallbackTop = *fallback.rbegin();
        if (fallbackTop != currH) {
            currH = fallbackTop;
            ans.emplace_back(std::vector<int>{p, currH});
        }
    }

    return ans;
}


// 2957. Remove Adjacent Almost-Equal Characters

int removeAlmostEqualCharacters(std::string word) {
    int count = 0;
    for (int i = 1; i < word.size(); ++i) {
        if (std::abs(word[i] - word[i - 1]) <= 1) {
            ++count;
            ++i;
        }
    }
    return count;
}

// 149. Max Points on a Line

std::string convertToStr(std::vector<int> &v1, std::vector<int> &v2) {
    return "p1:" + std::to_string(v1[0]) + ";" + std::to_string(v1[1]) + ";"
                                                                         "p2:" + std::to_string(v2[0]) + ";" +
           std::to_string(v2[1]);
}

bool isAligned(std::vector<int> &a, std::vector<int> &b, std::vector<int> &c) {
    // Calculate vectors formed by pairs of points (b-a) and (c-a)
    int v1_x = b[0] - a[0];
    int v1_y = b[1] - a[1];

    int v2_x = c[0] - a[0];
    int v2_y = c[1] - a[1];

    // Calculate the cross product
    int crossProduct = v1_x * v2_y - v1_y * v2_x;

    // If cross product is zero, points are collinear
    return crossProduct == 0;
}

int maxPoints(std::vector<std::vector<int>> &points) {
    // need used pairs
    // need mapping from used pair to count

    if (points.size() < 3) return points.size();

    int n = points.size();

    std::unordered_set < std::string > usedPairs;
    std::unordered_map<std::string, int> pointsCount;

    int maxCount = 0;

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            auto key = convertToStr(points[i], points[j]);

            if (usedPairs.contains(key)) continue;

            usedPairs.insert(key);

            pointsCount[key] = 2;
            for (int k = j + 1; k < n; ++k) {
                if (isAligned(points[i], points[j], points[k])) {
                    ++pointsCount[key];
                }
            }

            maxCount = std::max(maxCount, pointsCount[key]);
        }
    }

    return maxCount;
}

// 54. Spiral Matrix

std::vector<int> spiralOrder(std::vector<std::vector<int>> &matrix) {
    int n = matrix.size();
    int m = matrix[0].size();

    int bl = 0;
    int br = m - 1;

    int bt = 1;
    int bb = n - 1;

    int i = 0;
    int j = -1;

    int count = n * m;

    std::vector<int> res;
    while (count > 0) {
        for (j = j + 1; j <= br && count > 0; ++j) {
            res.emplace_back(matrix[i][j]);
            --count;
        }
        --br;
        --j;

        for (i = i + 1; i <= bb && count > 0; ++i) {
            res.emplace_back(matrix[i][j]);
            --count;
        }
        --bb;
        --i;

        for (j = j - 1; j >= bl && count > 0; --j) {
            res.emplace_back(matrix[i][j]);
            --count;
        }
        ++bl;
        ++j;

        for (i = i - 1; i >= bt && count > 0; --i) {
            res.emplace_back(matrix[i][j]);
            --count;
        }
        ++bt;
        ++i;
    }

    return res;
}

// 722. Remove Comments

std::vector<std::string> removeComments(std::vector<std::string> &source) {
    std::vector<std::string> ans;
    std::string s;
    bool comment = false;
    for (auto line: source) {
        int n = line.size();
        for (int j = 0; j < line.size(); j++) {
            if (!comment && j + 1 < n && line[j] == '/' && line[j + 1] == '/') break;
            else if (!comment && j + 1 < n && line[j] == '/' && line[j + 1] == '*') comment = true, j++;
            else if (comment && j + 1 < n && line[j] == '*' && line[j + 1] == '/') comment = false, j++;
            else if (!comment) s.push_back(line[j]);
        }

        if (!comment && s.size()) ans.push_back(s), s.clear();
    }
    return ans;
}

// 454. 4Sum II

int fourSumCount(std::vector<int> &nums1, std::vector<int> &nums2, std::vector<int> &nums3, std::vector<int> &nums4) {
    int ans = 0;
    std::unordered_map<int, int> sum;

    for (int k = 0; k < nums3.size(); ++k) {
        for (int t = 0; t < nums4.size(); ++t) {
            ++sum[nums3[k] + nums4[t]];
        }
    }

    for (int i = 0; i < nums1.size(); ++i) {
        for (int j = 0; j < nums2.size(); ++j) {
            ans += sum[-(nums1[i] + nums2[j])];
        }
    }

    return ans;
}

// 334. Increasing Triplet Subsequence

bool increasingTriplet(std::vector<int> &nums) {
    int i = 0;
    int j = 1;

    for (j = 1; j < nums.size() && nums[j] <= nums[i]; ++j) {
        if (nums[j] < nums[i]) {
            i = j;
        }
    }

    for (int k = j + 1; k < nums.size(); ++k) {
        if (nums[k] < nums[i]) {
            i = k;
        }
        if (nums[k] > nums[i] && nums[k] < nums[j]) {
            j = k;
        }
        if (nums[k] > nums[j]) {
            return true;
        }
    }

    return false;
}

// 11. Container With Most Water

int maxArea(std::vector<int> &height) {
    int l = 0;
    int r = height.size() - 1;

    int maxAr = 0;
    while (l < r) {
        maxAr = std::max(maxAr, (r - l) * std::min(height[l], height[r]));
        if (height[l] < height[r]) {
            ++l;
        } else {
            --r;
        }
    }

    return maxAr;
}

// 48. Rotate Image

void rotate(std::vector<std::vector<int>> &matrix) {
    int n = matrix.size();
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n - i - 1; ++j) {
            // {i, j},
            std::vector<std::pair<int, int>> dirs = {{j,         n - 1 - i},
                                                     {n - 1 - i, n - 1 - j},
                                                     {n - 1 - j, i},
                                                     {i,         j}};
            int carry = matrix[i][j];
            for (auto [di, dj]: dirs) {
                std::swap(matrix[di][dj], carry);
            }
        }
    }
}

// 46. Permutations

bool nextPermutation(vector<int> &nums) {
    int n = nums.size();

    int pivot = n - 2;
    while (pivot >= 0 && nums[pivot] >= nums[pivot + 1]) --pivot;

    if (pivot < 0) return false;

    // min element > nums[pivot]
    int i = n - 1;
    while (i >= 0 && nums[i] <= nums[pivot]) --i;

    std::swap(nums[i], nums[pivot]);

    std::reverse(nums.begin() + pivot + 1, nums.end());

    return true;
}

std::vector<std::vector<int>> permute(std::vector<int> &nums) {
    if (nums.size() == 1) return std::vector<std::vector<int>>{nums};

    std::sort(nums.begin(), nums.end());

    std::vector<std::vector<int>> res;
    res.emplace_back(std::vector<int>(nums.begin(), nums.end()));

    while (nextPermutation(nums)) {
        res.emplace_back(std::vector<int>(nums.begin(), nums.end()));
    }

    return res;
}

// 1094. Car Pooling

bool carPooling(std::vector<std::vector<int>> &trips, int capacity) {
    std::vector<std::pair<int, int>> path;

    for (auto trip: trips) {
        int numPass = trip[0];
        int from = trip[1];
        int to = trip[2];

        path.emplace_back(std::pair<int, int>{from, numPass});
        path.emplace_back(std::pair<int, int>{to, -numPass});
    }

    std::sort(path.begin(), path.end());

    for (auto [dir, numPass]: path) {
        if (numPass < 0) {
            // this adds capacity
            capacity -= numPass;
        } else if (capacity - numPass < 0) {
            return false;
        } else {
            // this decreases capacity
            capacity -= numPass;
        }
    }

    return true;
}

// 1156. Swap For Longest Repeated Character Substring

int maxRepOpt1(std::string s) {
    int n = s.size();

    std::unordered_map<char, std::vector<int>> groups;
    for (int i = 0; i < n; ++i) {
        groups[s[i]].emplace_back(i);
    }

    int res = 0;
    for (auto [g, idxs] : groups) {
        int prev = 0;
        int curr = 1;
        int ans = 0;

        for (int i = 1; i < idxs.size(); ++i) {
            if (idxs[i] == idxs[i - 1] + 1) ++curr;
            else {
                if (idxs[i] == idxs[i - 1] + 2) {
                    prev = curr;
                } else {
                    prev = 0;
                }
                curr = 1;
            }
            ans = std::max(ans, prev + curr);
        }
        res = std::max(res, ans + (idxs.size() > ans));
    }

    return res;
}

// 2405. Optimal Partition of String

int partitionString(std::string s) {
    int n = s.length();

    int used = 0;

    int count = 1;
    int l = 0;
    for (int r = 0; r < n; ++r) {
        int i = s[r] - 'a';
        if (used & (1 << i)) {
            ++count;
            l = r;
            used = 0;
        }
        used = used | (1 << i);
    }

    return count;
}

// 1878. Get Biggest Three Rhombus Sums in a Grid

void pop_min(std::set<int> &s) {
    auto itr = s.begin();
    s.erase(itr);
}

void exploreRhombus(
        int i, int j,
        std::vector<std::vector<int>> &grid,
        std::set<int> &s,
        std::vector<std::vector<int>> &ld,
        std::vector<std::vector<int>> &rd
) {
    int n = grid.size();
    int m = grid[0].size();

    for (int len = 1; len <= std::max(n, m); ++len) {
        int left = j - len;
        int right = j + len;
        int bot = i + 2 * len;

        if(left < 0 || right >= m || bot >= n) continue;

        int rhomb = rd[i + len][left] - rd[i][j]
                    + ld[i + len][right] - ld[i][j]
                    + ld[bot][j] - ld[i + len][left]
                    + rd[bot][j] - rd[i + len][right]
                    + grid[i][j] - grid[bot][j];

        s.insert(rhomb);
        if (s.size() > 3) pop_min(s);
    }
}

std::vector<int> getBiggestThree(std::vector<std::vector<int>>& grid) {
    int n = grid.size();
    int m = grid[0].size();

    std::vector<std::vector<int> > ld = grid, rd = grid;

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            int pi = i - 1;
            int prevj = j - 1;
            if(pi >= 0 && prevj >= 0) ld[i][j] += ld[pi][prevj];
            prevj = j + 1;
            if(pi >= 0 && prevj < m) rd[i][j] += rd[pi][prevj];
        }
    }

    std::set<int> s;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            s.insert(grid[i][j]);
            if (s.size() > 3) pop_min(s);

            exploreRhombus(i, j, grid, s, ld, rd);
        }
    }

    std::vector<int> res;
    for (auto el : s) {
        res.emplace_back(el);
    }

    std::reverse(res.begin(), res.end());

    return res;
}

// 2260. Minimum Consecutive Cards to Pick Up

int minimumCardPickup(std::vector<int>& cards) {
    int n = cards.size();
    int l = 0;

    int len = n + 1;

    std::unordered_map<int, int> prev;

    for (int i = 0; i < n; ++i) {
        if (prev.contains(cards[i])) {
            len = std::min(len, i - prev[cards[i]] + 1);
        }
        prev[cards[i]] = i;
    }

    if (len == n + 1) return -1;

    return len;
}

// 2439. Minimize Maximum of Array

int minimizeArrayValue(std::vector<int>& nums) {
    int n = nums.size();

    long long prefix_sum = nums[0];
    long long ans = nums[0];
    for (int i = 1; i < n; ++i) {
        prefix_sum += nums[i];
        ans = std::max(ans, (prefix_sum + i) / (i + 1));
    }

    return ans;
}

int main() {
    auto v = std::vector<std::vector<int>>{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
    };
    rotate(v);
    return 0;
}