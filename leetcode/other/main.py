# Brace Expansion II

class Solution:
    def findNext(self, expression: str, start: int):
        balance = 0
        for i, c in enumerate(expression[start:]):
            if c == '{':
                balance += 1
            elif c == '}':
                balance -= 1
                if balance == 0:
                    return start + i

        return -1

    def backtrack(self, expression: str) -> Set[str]:
        res = set()

        pos = 0
        curr_expr = {''}
        while pos < len(expression):
            c = expression[pos]

            if c not in '{},':
                curr_expr = {expr + c for expr in curr_expr}
            elif c == ',':
                res = res | curr_expr
                curr_expr = {''}
            elif c == '{':
                old_pos = pos + 1
                pos = self.findNext(expression, pos)

                # pos - 1 because we find next "}", so we don't need to include it
                sub_expr = self.backtrack(expression[old_pos:pos])

                tmp_expr = set()
                for expr1 in curr_expr:
                    for expr2 in sub_expr:
                        tmp_expr.add(expr1 + expr2)

                curr_expr = tmp_expr
            pos += 1

        res = res | curr_expr
        return res

    def braceExpansionII(self, expression: str) -> List[str]:
        return sorted(self.backtrack(expression))