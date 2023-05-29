from typing import List

class csv:
    def __init__(self, length: int = None):
        self.maxview: int = 4
        self.labels: List[str]
        self.data: List[List[str]]
        if not length is None:
            self.labels = []
            self.data = [[] for _ in range(length)]

    @classmethod
    def read(cls, fname: str, sep: str = ',', encoding = 'utf-8'):
        ret = cls()
        with open(fname, 'r', encoding=encoding) as f:
            ret.labels = [i.strip() for i in f.readline().strip().split(sep)]
            ret.data = [[j.strip() for j in i.strip().split(sep)] for i in f.readlines() if i.strip()]
        return ret

    def __str__(self) -> str:
        col_lenmax = [len(i) for i in self.labels]
        for line in self.data:
            for i, elem in enumerate(line):
                col_lenmax[i] = max(col_lenmax[i], len(elem))
        ret = ['|' + '|'.join(elem + ' ' * (col_lenmax[i] - len(elem)) for i, elem in enumerate(self.labels)) + '|', '|' + '|'.join('-' * i for i in col_lenmax) + '|']
        if len(self.data) <= self.maxview * 2:
            for line in self.data:
                ret.append('|' + '|'.join(elem + ' ' * (col_lenmax[i] - len(elem)) for i, elem in enumerate(line)) + '|')
        else:
            for line in self.data[:self.maxview]:
                ret.append('|' + '|'.join(elem + ' ' * (col_lenmax[i] - len(elem)) for i, elem in enumerate(line)) + '|')
            ret.append('|' + '|'.join('...' + ' ' * (col_lenmax[i] - 3) for i in range(len(self.labels))) + '|')
            for line in self.data[-self.maxview:]:
                ret.append('|' + '|'.join(elem + ' ' * (col_lenmax[i] - len(elem)) for i, elem in enumerate(line)) + '|')
        return '\n'.join(ret)

    def write(self, fname: str, sep: str = ',', encoding = 'utf-8'):
        with open(fname, 'w', encoding=encoding) as f:
            f.write(sep.join(self.labels))
            f.write('\n')
            f.writelines(sep.join(line) + '\n' for line in self.data)

    def __getitem__(self, col_name: str):
        idx = self.labels.index(col_name)
        return [line[idx] for line in self.data]
    
    def __setitem__(self, col_name: str, obj: List[str]):
        if col_name in self.labels:
            idx = self.labels.index(col_name)
            for line, val in zip(self.data, obj):
                line[idx] = val
        else:
            self.labels.append(col_name)
            for line, val in zip(self.data, obj):
                line.append(val)
    
    def __len__(self):
        return len(self.data)
    
    @property
    def shape(self):
        return len(self.data), len(self.labels)