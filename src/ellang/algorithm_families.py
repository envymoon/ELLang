from __future__ import annotations

import heapq
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class FamilyExecution:
    family: str
    task: str
    result: Any


REGISTRY: dict[tuple[str, str], Any] = {}


def execute_algorithm_family(family: str, task: str, bindings: dict[str, Any]) -> FamilyExecution:
    solver = REGISTRY.get((family, task))
    if solver is None:
        raise ValueError(f"Unsupported algorithm family task: {family}/{task}")
    return FamilyExecution(family=family, task=task, result=solver(bindings))


def is_supported_algorithm_task(family: str, task: str) -> bool:
    return (family, task) in REGISTRY


def supported_algorithm_tasks() -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for family, task in REGISTRY:
        families.setdefault(family, []).append(task)
    return {family: sorted(tasks) for family, tasks in sorted(families.items())}


def _three_sum(bindings: dict[str, Any]) -> list[list[int]]:
    nums = sorted(int(item) for item in bindings.get("nums", []))
    result: list[list[int]] = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1
    return result


def _rotate_array(bindings: dict[str, Any]) -> list[Any]:
    nums = list(bindings.get("nums", []))
    if not nums:
        return []
    steps = int(bindings.get("k", 0))
    offset = steps % len(nums)
    if offset == 0:
        return nums
    return nums[-offset:] + nums[:-offset]


def _binary_search(bindings: dict[str, Any]) -> int:
    nums = [int(item) for item in bindings.get("nums", [])]
    target = int(bindings.get("target", 0))
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


def _max_subarray(bindings: dict[str, Any]) -> int:
    nums = [int(item) for item in bindings.get("nums", [])]
    if not nums:
        return 0
    best = current = nums[0]
    for value in nums[1:]:
        current = max(value, current + value)
        best = max(best, current)
    return best


def _product_except_self(bindings: dict[str, Any]) -> list[int]:
    nums = [int(item) for item in bindings.get("nums", [])]
    if not nums:
        return []
    prefix = [1] * len(nums)
    suffix = [1] * len(nums)
    for index in range(1, len(nums)):
        prefix[index] = prefix[index - 1] * nums[index - 1]
    for index in range(len(nums) - 2, -1, -1):
        suffix[index] = suffix[index + 1] * nums[index + 1]
    return [prefix[index] * suffix[index] for index in range(len(nums))]


def _group_anagrams(bindings: dict[str, Any]) -> list[list[str]]:
    words = [str(item) for item in bindings.get("words", [])]
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for word in words:
        groups[tuple(sorted(word))].append(word)
    return list(groups.values())


def _most_frequent_element(bindings: dict[str, Any]) -> Any:
    values = list(bindings.get("elements", bindings.get("nums", [])))
    if not values:
        return None
    counts = Counter(values)
    best_count = max(counts.values())
    for value in values:
        if counts[value] == best_count:
            return value
    return None


def _valid_parentheses(bindings: dict[str, Any]) -> bool:
    text = str(bindings.get("s", ""))
    stack: list[str] = []
    pairs = {")": "(",
        "]": "[",
        "}": "{",
    }
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack


def _top_k_frequent(bindings: dict[str, Any]) -> list[int]:
    nums = [int(item) for item in bindings.get("nums", [])]
    k = int(bindings.get("k", 0))
    return [item for item, _ in Counter(nums).most_common(k)]


def _reverse_list(bindings: dict[str, Any]) -> list[Any]:
    values = list(bindings.get("head", []))
    return list(reversed(values))


def _binary_tree_level_order(bindings: dict[str, Any]) -> list[list[Any]]:
    nodes = list(bindings.get("root", []))
    if not nodes:
        return []
    result: list[list[Any]] = []
    queue: deque[tuple[int, int]] = deque([(0, 0)])
    while queue:
        index, depth = queue.popleft()
        if index >= len(nodes) or nodes[index] is None:
            continue
        if len(result) <= depth:
            result.append([])
        result[depth].append(nodes[index])
        queue.append((2 * index + 1, depth + 1))
        queue.append((2 * index + 2, depth + 1))
    return result


def _num_islands(bindings: dict[str, Any]) -> int:
    grid = [list(map(str, row)) for row in bindings.get("grid", [])]
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    seen: set[tuple[int, int]] = set()
    islands = 0

    def bfs(sr: int, sc: int) -> None:
        queue: deque[tuple[int, int]] = deque([(sr, sc)])
        seen.add((sr, sc))
        while queue:
            r, c = queue.popleft()
            for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen and grid[nr][nc] == "1":
                    seen.add((nr, nc))
                    queue.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r, c) not in seen:
                islands += 1
                bfs(r, c)
    return islands


def _binary_tree_right_side_view(bindings: dict[str, Any]) -> list[Any]:
    levels = _binary_tree_level_order(bindings)
    return [level[-1] for level in levels if level]


def _coin_change(bindings: dict[str, Any]) -> int:
    coins = [int(item) for item in bindings.get("coins", [])]
    amount = int(bindings.get("amount", 0))
    dp = [amount + 1] * (amount + 1)
    dp[0] = 0
    for value in range(1, amount + 1):
        for coin in coins:
            if coin <= value:
                dp[value] = min(dp[value], dp[value - coin] + 1)
    return -1 if dp[amount] > amount else dp[amount]


def _subsets(bindings: dict[str, Any]) -> list[list[Any]]:
    nums = list(bindings.get("nums", []))
    result: list[list[Any]] = [[]]
    for value in nums:
        result.extend([subset + [value] for subset in result])
    return result


def _longest_increasing_subsequence(bindings: dict[str, Any]) -> int:
    nums = [int(item) for item in bindings.get("nums", [])]
    if not nums:
        return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


REGISTRY.update(
    {
        ("array_manipulation", "rotate_array"): _rotate_array,
        ("array_manipulation", "binary_search"): _binary_search,
        ("array_manipulation", "max_subarray"): _max_subarray,
        ("array_manipulation", "product_except_self"): _product_except_self,
        ("array_two_pointers", "three_sum"): _three_sum,
        ("hashmap_counting", "group_anagrams"): _group_anagrams,
        ("hashmap_counting", "most_frequent_element"): _most_frequent_element,
        ("stack_queue_heap", "valid_parentheses"): _valid_parentheses,
        ("stack_queue_heap", "top_k_frequent"): _top_k_frequent,
        ("linked_list", "reverse_list"): _reverse_list,
        ("tree_graph", "binary_tree_level_order"): _binary_tree_level_order,
        ("tree_graph", "binary_tree_right_side_view"): _binary_tree_right_side_view,
        ("tree_graph", "num_islands"): _num_islands,
        ("dp_backtracking", "coin_change"): _coin_change,
        ("dp_backtracking", "subsets"): _subsets,
        ("dp_backtracking", "longest_increasing_subsequence"): _longest_increasing_subsequence,
    }
)
