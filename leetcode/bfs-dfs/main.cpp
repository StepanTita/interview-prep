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

// 838. Push Dominoes

std::string pushDominoes(std::string dominoes) {
    std::queue<int> q;

    int n = dominoes.length();

    for (int i = 0; i < n; ++i) {
        if (dominoes[i] != '.') {
            q.push(i);
        }
    }

    while (!q.empty()) {
        auto curr = q.front();
        q.pop();

        if (dominoes[curr] == 'L') {
            if (curr - 1 >= 0 && dominoes[curr - 1] == '.') {
                dominoes[curr - 1] = 'L';
                q.push(curr - 1);
            }
        } else {
            if (curr + 1 < n && dominoes[curr + 1] == '.') {
                if (curr + 2 >= n || dominoes[curr + 2] != 'L') {
                    dominoes[curr + 1] = 'R';
                    q.push(curr + 1);
                } else if (curr + 2 < n) {
                    q.pop();
                }
            }
        }
    }

    return dominoes;
}

// 854. K-Similar Strings

int kSimilarity(std::string s1, std::string s2) {
    int n = s1.length();

    std::queue<std::string> q;
    q.push(s1);

    std::unordered_map<std::string, int> dist;
    dist[s1] = 0;

    while (!q.empty()) {
        auto curr = q.front();
        q.pop();

        int i = 0;
        while (i < n && curr[i] == s2[i]) ++i;

        for (int j = i + 1; j < n; ++j) {
            if (curr[j] == s2[i] && s2[j] != curr[j]) {
                std::string next = curr;
                std::swap(next[i], next[j]);

                if (dist.contains(next)) continue;

                q.push(next);

                dist[next] = dist[curr] + 1;

                if (next == s2) return dist[next];
            }
        }
    }

    return dist[s2];
}