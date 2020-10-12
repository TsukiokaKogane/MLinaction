from data2 import rawdata, domain, attr


class CART(object):
    def __init__(self, d, a):
        self.edge = []
        self.node = []
        self.domain = d
        self.attr = a

    def get_gini(self, d1, d2, data, classify):
        cnt = [[0] * len(self.domain[classify]) for j in range(2)]
        tot = 0
        for d in data:
            tot += d[0]
            if self.domain[d1].index(d[d1]) == d2:
                cnt[1][self.domain[classify].index(d[classify])] += d[0]
            else:
                cnt[0][self.domain[classify].index(d[classify])] += d[0]
        res = 0.0
        for i in range(2):
            temp = 1.0
            for j in range(len(self.domain[classify])):
                if sum(cnt[i]) > 0:
                    temp -= (cnt[i][j] / sum(cnt[i])) ** 2
            res += temp * (sum(cnt[i]) / tot)
        return res

    def generate(self, data, classify):
        flag = 0
        temp = []
        num = len(self.node)
        # print('num=' + str(num))
        # self.node.append([])
        self.edge.append([])
        for d in data:
            if d[classify] not in temp:
                if len(temp) == 0:
                    temp.append(d[classify])
                else:
                    flag = 1
                    break
        if flag == 0:
            # print(temp)
            self.node.append(temp[0])
            return
        candidate_list = []
        for i in range(1, len(self.domain)):
            if i != classify:
                for j in range(len(self.domain[i])):
                    candidate_list.append((i, j, self.get_gini(i, j, data, classify)))
        sorted_candidate_list = sorted(candidate_list, key=lambda x: x[2])
        # print(sorted_candidate_list)

        picked_attr = sorted_candidate_list[0][0]
        picked_attr_value = sorted_candidate_list[0][1]
        # print('picked_attr= ' + str(self.attr[picked_attr]))
        # print('picked_attr_value= ' + self.domain[picked_attr][picked_attr_value])
        data0 = []
        data1 = []
        self.node.append(self.attr[picked_attr])
        # print(self.node[num])
        for d in data:
            if d[picked_attr] == self.domain[picked_attr][picked_attr_value]:
                data1.append(d)
            else:
                data0.append(d)

        self.edge[num].append(
            (num, len(self.node), 'not_' + self.domain[picked_attr][picked_attr_value]))
        self.generate(data0, classify)

        self.edge[num].append(
            (num, len(self.node), self.domain[picked_attr][picked_attr_value]))
        self.generate(data1, classify)

    def dfs(self, node):
        for e in self.edge[node]:
            print(self.node[e[0]] + ' ' + e[2] + ' ' + self.node[e[1]])
            self.dfs(e[1])


if __name__ == '__main__':
    c = CART(domain, attr)
    c.generate(data=rawdata, classify=5)
    c.dfs(0)

