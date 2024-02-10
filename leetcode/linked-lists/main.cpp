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

struct ListNode {
    int val;
    ListNode *next;

    ListNode() : val(0), next(nullptr) {}

    ListNode(int x) : val(x), next(nullptr) {}

    ListNode(int x, ListNode *next) : val(x), next(next) {}
};


ListNode *removeNthFromEnd(ListNode *head, int n) {
    auto fast = head;
    auto slow = head;

    for (int i = 0; i < n; ++i) {
        fast = fast->next;
    }

    if (fast == NULL) return head->next;

    while (fast->next) {
        fast = fast->next;
        slow = slow->next;
    }

    slow->next = slow->next->next;

    return head;
}

// 143. Reorder List

void reorderList(ListNode *head) {
    if (head->next == NULL) return;

    ListNode *prev = NULL;
    auto slow = head;
    auto fast = head;

    while (fast != NULL && fast->next != NULL) {
        prev = slow;
        slow = slow->next;
        fast = fast->next->next;
    }

    if (prev != NULL) {
        prev->next = NULL;
    }

    // slow - is the first element of the second half
    // reverse slow

    prev = NULL;
    auto curr = slow;

    while (curr != NULL) {
        auto next = curr->next;
        curr->next = prev;
        prev = curr;

        curr = next;
    }

    auto a = head;
    auto b = prev;

    auto res = new ListNode();
    curr = res;

    while (a != NULL && b != NULL) {
        auto nextA = a->next;
        auto nextB = b->next;

        curr->next = a;
        curr->next->next = b;

        a = nextA;
        b = nextB;

        curr = curr->next->next;
    }

    while (a != NULL) {
        auto nextA = a->next;
        curr->next = a;
        a = nextA;

        curr = curr->next;
    }

    while (b != NULL) {
        auto nextB = b->next;
        curr->next = b;
        b = nextB;

        curr = curr->next;
    }

}

int main() {
    auto l = new ListNode(1);
    reorderList(l);
    return 0;
}