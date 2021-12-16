"""
aoc.py

Advent of Code 2021 solutions.
"""

import math
import re
from collections import Counter, defaultdict
from importlib import resources


def day_1(days_to_compare: int = 1) -> int:
    """Day 1 Problem 1."""
    with resources.open_text("advent_of_code.data", "day_1.txt") as raw_text:
        data = [int(x) for x in raw_text.read().splitlines()]
    return sum((x > data[i] for i, x in enumerate(data[days_to_compare:])))


def day_1_2() -> int:
    """Day 1 Problem 2."""
    return day_1(3)


def load_day_2_moves() -> list[tuple[str, int]]:
    """Load moves for Day 2 functions."""
    moves = []
    with resources.open_text("advent_of_code.data", "day_2.txt") as raw_text:
        for line in raw_text.read().splitlines():
            direction, str_delta = line.split()
            moves.append((direction, int(str_delta)))
    return moves


def day_2() -> int:
    """Day 2 Problem 1."""
    moves = load_day_2_moves()
    horizontal = 0
    depth = 0
    for direction, delta in moves:
        if direction == "up":
            depth -= delta
        elif direction == "down":
            depth += delta
        elif direction == "forward":
            horizontal += delta
        else:
            raise ValueError
    return horizontal * depth


def day_2_2() -> int:
    """Day 2 Problem 2."""
    moves = load_day_2_moves()
    aim = 0
    horizontal = 0
    depth = 0
    for direction, delta in moves:
        if direction == "up":
            aim -= delta
        elif direction == "down":
            aim += delta
        elif direction == "forward":
            horizontal += delta
            depth += delta * aim
        else:
            raise ValueError
    return horizontal * depth


def day_3() -> int:
    """Day 3 Problem 1."""
    with resources.open_text("advent_of_code.data", "day_3.txt") as raw_text:
        data = raw_text.read().splitlines()
    gamma_bits = []
    for pos in range(len(data[0])):
        bits = [int(line[pos]) for line in data]
        gamma_bits.append(Counter(bits).most_common(1)[0][0])
    epsilon_bits = [(b - 1) % 2 for b in gamma_bits]
    gamma = int("".join([str(i) for i in gamma_bits]), 2)
    epsilon = int("".join([str(i) for i in epsilon_bits]), 2)
    return gamma * epsilon


def day_3_2() -> int:
    """Day 3 Problem 2."""
    with resources.open_text("advent_of_code.data", "day_3.txt") as raw_text:
        data = raw_text.read().splitlines()
    digits_len = len(data[0])

    def filter_data(
        data: list[str], pos: int, least_common: bool = False
    ) -> list[str]:
        data_len = len(data)
        if least_common:
            zeros = sum(1 for row in data if row[pos] == "0")
            value = str(int(2 * zeros > data_len))
        else:
            ones = sum(1 for row in data if row[pos] == "1")
            value = str(int(2 * ones >= data_len))
        return [row for row in data if row[pos] == value]

    def get_rating(data: list[str], least_common: bool = False) -> int:
        for i in range(digits_len):
            data = filter_data(data, i, least_common)
            if len(data) == 1:
                return int("".join([str(i) for i in data[0]]), 2)
        raise ValueError

    oxygen = get_rating(data)
    carbon_dioxide = get_rating(data, least_common=True)
    return oxygen * carbon_dioxide


def day_4_parser() -> tuple[
    list[str],
    dict[str, int],
    list[dict[tuple[int, int], str]],
    list[dict[tuple[int, int], int]],
]:
    """Parser for Day 4 Data File."""
    with resources.open_text("advent_of_code.data", "day_4.txt") as raw_text:
        lines = raw_text.readlines()
        calls = lines[0].strip().split(",")
        called = {call: 0 for call in calls}
        cards = []
        states = []
        for i, line in enumerate(lines[1:]):
            if i % 6 == 0:
                card = {}
            else:
                for j, value in enumerate(line.split()):
                    card[i % 6 - 1, j] = value
                if i % 6 == 5:
                    cards.append(card)
                    states.append(
                        {(i, j): 0 for i in range(5) for j in range(5)}
                    )
        return calls, called, cards, states


def day_4() -> int:
    """Day 4 Problem 1."""
    calls, called, cards, states = day_4_parser()

    for call in calls:
        called[call] = 1
        for index, card in enumerate(cards):
            state = states[index]
            for key, value in card.items():
                state[key] = called[value]
            if (
                max(sum(state[i, j] for j in range(5)) for i in range(5)) == 5
            ) or (
                max(sum(state[i, j] for i in range(5)) for j in range(5)) == 5
            ):
                score = sum(
                    int(v) for k, v in card.items() if state[k] == 0
                ) * int(call)
                return score

    raise ValueError


def day_4_2() -> int:
    """Day 4 Problem 2."""
    calls, called, cards, states = day_4_parser()
    win_statuses = [False for card in cards]

    for call in calls:
        called[call] = 1
        for index, card in enumerate(cards):
            state = states[index]
            for key, value in card.items():
                state[key] = called[value]
            if (
                max(sum(state[i, j] for j in range(5)) for i in range(5)) == 5
            ) or (
                max(sum(state[i, j] for i in range(5)) for j in range(5)) == 5
            ):
                win_statuses[index] = True
            if all(win_statuses):
                score = sum(
                    int(v) for k, v in card.items() if state[k] == 0
                ) * int(call)
                return score

    raise ValueError


def day_5() -> int:
    """Day 5 Problem 1."""
    with resources.open_text("advent_of_code.data", "day_5.txt") as raw_text:
        coords = [
            [int(i) for i in re.split(r"\D+", line.strip())]
            for line in raw_text
        ]

    scores: defaultdict[tuple[int, int], int] = defaultdict(int)
    for row in coords:
        if row[0] == row[2]:
            min_ = min(row[1], row[3])
            max_ = max(row[1], row[3])
            for point in range(min_, max_ + 1):
                scores[row[0], point] += 1
        elif row[1] == row[3]:
            min_ = min(row[0], row[2])
            max_ = max(row[0], row[2])
            for point in range(min_, max_ + 1):
                scores[point, row[1]] += 1

    return sum(1 for s in scores.values() if s > 1)


def day_5_2() -> int:
    """Day 5 Problem 2."""
    with resources.open_text("advent_of_code.data", "day_5.txt") as raw_text:
        coords = [
            [int(i) for i in re.split(r"\D+", line.strip())]
            for line in raw_text
        ]

    scores: defaultdict[tuple[int, int], int] = defaultdict(int)
    for row in coords:
        if row[0] == row[2]:
            min_ = min(row[1], row[3])
            max_ = max(row[1], row[3])
            for point in range(min_, max_ + 1):
                scores[row[0], point] += 1
        elif row[1] == row[3]:
            min_ = min(row[0], row[2])
            max_ = max(row[0], row[2])
            for point in range(min_, max_ + 1):
                scores[point, row[1]] += 1
        elif abs(row[0] - row[2]) == abs(row[1] - row[3]):
            left = (row[0], row[1]) if row[0] <= row[2] else (row[2], row[3])
            slope = (row[3] - row[1]) // (row[2] - row[0])
            for offset in range(abs(row[0] - row[2]) + 1):
                scores[left[0] + offset, left[1] + offset * slope] += 1
        else:
            raise ValueError
    return sum(1 for s in scores.values() if s > 1)


def day_6(days: int = 80) -> int:
    """Day 6 Problem 1."""
    with resources.open_text("advent_of_code.data", "day_6.txt") as raw_text:
        population = [int(i) for i in raw_text.readline().split(",")]
    grouped_pop = {i: 0 for i in range(9)}
    for fish in population:
        grouped_pop[fish] += 1
    for _ in range(days):
        grouped_pop_old = grouped_pop.copy()
        grouped_pop = {i: 0 for i in range(9)}
        for i in range(9):
            grouped_pop[i] = grouped_pop_old[(i + 1) % 9]
        grouped_pop[6] += grouped_pop_old[0]
    return sum(grouped_pop.values())


def day_6_2() -> int:
    """Day 6 Problem 2."""
    return day_6(256)


def day_7() -> int:
    """Day 7 Problem 1."""
    with resources.open_text("advent_of_code.data", "day_7.txt") as raw_text:
        crabs = [int(i) for i in raw_text.readline().split(",")]
    scores = {}

    for position in range(min(crabs), max(crabs) + 1):
        scores[position] = sum(abs(crab - position) for crab in crabs)
    return min(scores.values())


def day_7_2() -> int:
    """Day 7 Problem 2."""
    with resources.open_text("advent_of_code.data", "day_7.txt") as raw_text:
        crabs = [int(i) for i in raw_text.readline().split(",")]
    scores = {}

    for position in range(min(crabs), max(crabs) + 1):
        scores[position] = 0
        for crab in crabs:
            dist = abs(crab - position)
            scores[position] += dist * (dist + 1) // 2
    return min(scores.values())


def day_8() -> int:
    """Day 8 Problem 1."""
    with resources.open_text("advent_of_code.data", "day_8.txt") as raw_text:
        outputs = [line.strip().split(" ")[-4:] for line in raw_text]
    lens = (2, 3, 4, 7)
    return sum(1 for line in outputs for word in line if len(word) in lens)


def day_8_2() -> int:
    """Day 8 Problem 2."""
    numbers = {
        frozenset("abcefg"): "0",
        frozenset("cf"): "1",
        frozenset("acdeg"): "2",
        frozenset("acdfg"): "3",
        frozenset("bcdf"): "4",
        frozenset("abdfg"): "5",
        frozenset("abdefg"): "6",
        frozenset("acf"): "7",
        frozenset("abcdefg"): "8",
        frozenset("abcdfg"): "9",
    }

    with resources.open_text("advent_of_code.data", "day_8.txt") as raw_text:
        lines = [line.strip().split(" ") for line in raw_text.readlines()]

    total = 0
    for line in lines:
        input_ = line[:10]
        output = line[-4:]
        concatenated_input = "".join(input_)
        counter = Counter(concatenated_input)
        frequencies = {4: "e", 6: "b", 7: "d", 8: "c", 9: "f"}
        translations = {k: frequencies[v] for k, v in counter.items()}
        one = next(word for word in input_ if len(word) == 2)
        seven = next(word for word in input_ if len(word) == 3)
        four = next(word for word in input_ if len(word) == 4)
        to_a = next(letter for letter in seven if letter not in one)
        to_g = next(
            letter
            for letter in translations
            if (translations[letter] == "d" and letter not in four)
        )
        translations[to_a] = "a"
        translations[to_g] = "g"
        trans_out = [("".join(translations[i] for i in o)) for o in output]
        digits_out = [numbers[frozenset(i)] for i in trans_out]
        total += int("".join(digits_out))

    return total


def day_9_parser() -> list[list[int]]:
    """Input parser for Day 9 problems."""
    map_ = []
    with resources.open_text("advent_of_code.data", "day_9.txt") as raw_text:
        for line in raw_text:
            map_.append([int(i) for i in line.strip()])
    return map_


def day_9_find_sinks(map_: list[list[int]]) -> list[tuple[int, int]]:
    """Find sinks in model."""

    displacements = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    x_max = len(map_[0]) - 1
    y_max = len(map_) - 1

    def adjacent_values(x_pos: int, y_pos: int) -> list[int]:
        """Get adjacent values."""
        adj = []
        for (d_x, d_y) in displacements:
            x_adj = x_pos + d_x
            y_adj = y_pos + d_y
            if x_adj >= 0 and x_adj <= x_max and y_adj >= 0 and y_adj <= y_max:
                adj.append(map_[y_adj][x_adj])
        return adj

    sinks = []
    for y_pos, row in enumerate(map_):
        for x_pos, value in enumerate(row):
            if value < min(adjacent_values(x_pos, y_pos)):
                sinks.append((x_pos, y_pos))

    return sinks


def day_9() -> int:
    """Day 9 Problem 1."""
    map_ = day_9_parser()
    sinks = day_9_find_sinks(map_)
    return sum(map_[y_pos][x_pos] + 1 for x_pos, y_pos in sinks)


def day_9_2() -> int:
    """Day 9 Problem 2."""
    map_ = day_9_parser()
    sinks = day_9_find_sinks(map_)

    displacements = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    x_max = len(map_[0]) - 1
    y_max = len(map_) - 1

    sizes: dict[tuple[int, int], int] = {}
    for x_pos, y_pos in sinks:
        seen = [[False for point in row] for row in map_]
        seen[y_pos][x_pos] = True
        size = 1
        while True:
            last_size = size
            for y_1, row in enumerate(map_):
                for x_1, _ in enumerate(row):
                    for d_x, d_y in displacements:
                        x_2 = x_1 + d_x
                        y_2 = y_1 + d_y
                        if (
                            seen[y_1][x_1]
                            and x_2 >= 0
                            and x_2 <= x_max
                            and y_2 >= 0
                            and y_2 <= y_max
                            and map_[y_2][x_2] < 9
                            and not seen[y_2][x_2]
                        ):
                            size += 1
                            seen[y_2][x_2] = True
            if last_size == size:
                sizes[(x_pos, y_pos)] = size
                break

    return math.prod(sorted(sizes.values())[-3:])


def day_10_solver(part_2: bool) -> int:
    """Day 10 Solver."""
    with resources.open_text("advent_of_code.data", "day_10.txt") as raw_text:
        lines = [line.rstrip() for line in raw_text.readlines()]

    valid_pairs = {"{": "}", "[": "]", "(": ")", "<": ">"}
    error_scores = {")": 3, "]": 57, "}": 1197, ">": 25137}
    completion_scores = {")": 1, "]": 2, "}": 3, ">": 4}
    total_error_score = 0
    line_completion_scores = []

    for line in lines:
        error_score = 0
        completion_score = 0
        seen = []
        for char in line:
            if char in valid_pairs:  # we have an opening character
                seen.append(char)
            elif seen and valid_pairs[seen.pop()] != char:
                error_score = error_scores[char]
                break
        total_error_score += error_score
        if not error_score:
            while seen:
                completion_score *= 5
                completion_score += completion_scores[valid_pairs[seen.pop()]]
            line_completion_scores.append(completion_score)

    if part_2:
        n_completion_scores = len(line_completion_scores)
        return sorted(line_completion_scores)[n_completion_scores // 2]
    else:
        return total_error_score


def day_10() -> int:
    """Day 10 Problem 1."""
    return day_10_solver(part_2=False)


def day_10_2() -> int:
    """Day 10 Problem 2."""
    return day_10_solver(part_2=True)


def day_11_solver(part_2: bool) -> int:
    """Solver for Day 11 Problem."""
    steps = 100
    width = 0
    height = 0

    with resources.open_text("advent_of_code.data", "day_11.txt") as raw_text:
        energies = {}
        flashes = {}
        for j, line in enumerate(raw_text):
            height = j + 1
            for i, str_value in enumerate(line.strip()):
                energies[i, j] = int(str_value)
                flashes[i, j] = False
                width = i + 1

    total_flashes = 0
    displacements = [
        (0, 1),
        (0, -1),
        (1, 0),
        (1, 1),
        (1, -1),
        (-1, 0),
        (-1, 1),
        (-1, -1),
    ]

    step = 0
    while True:
        if step >= steps and not part_2:
            break
        step += 1
        flashes_this_step = 0

        for k in energies:
            energies[k] += 1

        while True:
            new_flashes = 0
            for i, j in energies:
                if energies[i, j] > 9 and not flashes[i, j]:
                    flashes[i, j] = True
                    new_flashes += 1
                    for d_i, d_j in displacements:
                        adj_i = i + d_i
                        adj_j = j + d_j
                        if (
                            adj_i >= 0
                            and adj_i < width
                            and adj_j >= 0
                            and adj_j < height
                        ):
                            energies[adj_i, adj_j] += 1
            if not new_flashes:
                break
            else:
                flashes_this_step += new_flashes

        if part_2 and all(flashes.values()):
            return step

        for k, flash in flashes.items():
            if flash:
                flashes[k] = False
                energies[k] = 0
        total_flashes += flashes_this_step

    return -1 if part_2 else total_flashes


def day_11() -> int:
    """Day 11 Part 1."""
    return day_11_solver(part_2=False)


def day_11_2() -> int:
    """Day 11 Part 2."""
    return day_11_solver(part_2=True)
