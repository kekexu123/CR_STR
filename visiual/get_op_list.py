# from nltk.metrics.distance import _edit_dist_backtrace
import operator

def _edit_dist_init(len1, len2):
    lev = []
    for i in range(len1):
        lev.append([0] * len2)  # initialize 2D array to zero
    for i in range(len1):
        lev[i][0] = i  # column 0: 0,1,2,3,4,...
    for j in range(len2):
        lev[0][j] = j  # row 0: 0,1,2,3,4,...
    return lev


def _edit_dist_step(lev, i, j, s1, s2, substitution_cost=1, transpositions=False):
    c1 = s1[i - 1]
    c2 = s2[j - 1]

    # skipping a character in s1
    a = lev[i - 1][j] + 1
    # skipping a character in s2
    b = lev[i][j - 1] + 1
    # substitution
    c = lev[i - 1][j - 1] + (substitution_cost if c1 != c2 else 0)

    # transposition
    d = c + 1  # never picked by default
    if transpositions and i > 1 and j > 1:
        if s1[i - 2] == c2 and s2[j - 2] == c1:
            d = lev[i - 2][j - 2] + 1

    # pick the cheapest
    lev[i][j] = min(a, b, c, d)


def edit_distance(s1, s2, substitution_cost=1, transpositions=False):
    # set up a 2-D array
    len1 = len(s1)
    len2 = len(s2)
    lev = _edit_dist_init(len1 + 1, len2 + 1)

    # iterate over the array
    for i in range(len1):
        for j in range(len2):
            _edit_dist_step(
                lev,
                i + 1,
                j + 1,
                s1,
                s2,
                substitution_cost=substitution_cost,
                transpositions=transpositions,
            )
    align_list = _edit_dist_backtrace(lev)
    return lev[len1][len2], align_list

def _edit_dist_backtrace(lev):
    i, j = len(lev) - 1, len(lev[0]) - 1
    alignment = [(i, j)]

    while (i, j) != (0, 0):
        directions = [
            (i - 1, j),  # skip s1
            (i, j - 1),  # skip s2
            (i - 1, j - 1),  # substitution
        ]

        direction_costs = (
            (lev[i][j] if (i >= 0 and j >= 0) else float("inf"), (i, j))
            for i, j in directions
        )
        _, (i, j) = min(direction_costs, key=operator.itemgetter(0))

        alignment.append((i, j))
    return list(reversed(alignment))


def get_op_seq(pred, gt):
    _, align_list = edit_distance(pred, gt, 2, True)
    edit_flag = []
    for k in range(1, len(align_list)):
        pre_i, pre_j = align_list[k - 1]
        i, j = align_list[k]

        if i == pre_i:  ## skip gt char, p need being inserted
            edit_flag.append('i')
        elif j == pre_j:  ## skip p char, p need being deleted
            edit_flag.append('d')
        else:
            if pred[i - 1] != gt[j - 1]:  ## subsitution
                edit_flag.append('s')
            else:  ## correct
                edit_flag.append('#')
    return ''.join(edit_flag)


if __name__ == '__main__':
    p = 'lasda'
    gt1 = 'lsda'
    gt2 = 'lbsda'
    gt3 = 'laaa'
    print(get_op_seq(p,gt1))
    print(get_op_seq(p,gt2))
    print(get_op_seq(p,gt3))

