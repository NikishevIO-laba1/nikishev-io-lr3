import yaml

def yaml_load(file):
    with open(file, 'r', encoding = 'utf8') as f: return yaml.safe_load(f.read())

variables = yaml_load('лингвистические переменные.yaml')
rules = yaml_load('продукционные правила.yaml')
#objects = yaml_load('объекты.yaml')

def get_value(piecewise: dict, x):
    cur = 0
    for k,v in piecewise.items():
        #print(k, x, cur,piecewise.keys())
        if k in ('значение', "следствие"): continue
        if k == x: return v
        if k > x:
            i1, i2 = list(piecewise.keys())[cur-1], list(piecewise.keys())[cur]
            if cur == 0: return list(piecewise.values())[0]
            #print('cur', cur,'; x',x, '; piecewise', piecewise, '; i1',i1, '; i2',i2, '; pw[i1]',piecewise[i1], '; pw[i2]',piecewise[i2])
            return piecewise[i1] + ((piecewise[i2] - piecewise[i1]) * ((x - i1) / (i2 - i1)))
        cur += 1
    return list(piecewise.values())[-1]

def get_output(rule, vars):

    membership_values = []

    # определяются степени истинности
    for ling, fuz in rule['если'].items():
        var = vars[ling]
        truth = var['значение']
        membership_values.append(get_value(var['нечёткие переменные'][fuz], truth))

    if rule['операция'] == 'AND': value = min(membership_values)
    if rule['операция'] == 'OR': value = max(membership_values)
    return value

def clean(piecewise): return {k:v for k,v in piecewise.items() if k not in ('значение', "следствие")}
def apply_min(piecewise:dict, value):
    graph = clean(piecewise)
    x = list(graph.keys())
    y = list(graph.values())
    xcopy = x.copy()
    ycopy = y.copy()
    for i in range(len(x)):
        if y[i] > value:
            if i>0 and y[i] > ycopy[i-1]:
                x[i] = x[i-1] + (x[i] - x[i-1]) * abs(value-y[i-1])/abs(y[i] - y[i-1])
                y[i] = value
            elif i>0 and i<len(x)-1 and y[i+1] < y[i]:
                x[i] = x[i] + (x[i+1] - x[i]) * abs(value-y[i+1])/abs(y[i] - y[i+1])
                y[i] = value
            else:
                y[i] = value
    # if x[-1] != xcopy[-1]:
    #     x.append(xcopy[-1])
    #     y.append(min(ycopy[-1], value))
        
    graph = {k:v for k,v in zip(x,y)}
    return graph


import matplotlib.pyplot as plt
import numpy as np
def apply_min(piecewise:dict, value):
    xvals = np.linspace(list(piecewise.keys())[0], list(piecewise.keys())[-1], 1000)
    graph = np.array([np.interp(i, list(piecewise.keys()), list(piecewise.values())) for i in xvals])
    graph = np.clip(graph, None, value)
    return {xvals[i]:graph[i] for i in range(len(xvals))}

def update_graph(g1:dict, g2:dict):
    x1 = np.linspace(min(list(g1.keys())[0], list(g2.keys())[0]), max(list(g2.keys())[-1], list(g1.keys())[-1]), 1000)
    graph1 = np.array([np.interp(i, list(g1.keys()), list(g1.values())) for i in x1])
    graph2 = np.array([np.interp(i, list(g2.keys()), list(g2.values())) for i in x1])
    graphmin = np.maximum(graph1, graph2)
    return {x1[i]:graphmin[i] for i in range(len(x1))}

    
class Fuzzy:
    def __init__(self, rules:dict, objects:dict, variables:dict, input_vars:dict):
        self.rules = rules
        self.objects = objects
        self.variables = variables

        for varid, var in input_vars.items():
            self.variables[varid]['значение'] = var
            
    def plot_graphs(self):
        for varid, var in self.variables.items():
            for fuzid, fuz in var['нечёткие переменные'].items():
                graph = clean(fuz)
                plt.plot(list(graph.keys()), list(graph.values()), label = f'{fuzid}')
            plt.title(f'{varid}')
            plt.xlabel('y')
            plt.ylabel('μ(y)')
            plt.legend()
            plt.show()
            
    def fuzzificate(self):
        """На этапе физификации необходимо вывести степени истинности всех термов лингвистических переменных (в убывающем порядке степени истинности)."""
        truth_degrees = {}
        for varid, var in self.variables.items():
            if 'значение' in var:
                value = var['значение']
                for fuzid, fuz in var['нечёткие переменные'].items():
                    truth_degree = get_value(fuz, value)
                    var['нечёткие переменные'][fuzid]['значение'] = truth_degree
                    truth_degrees[f'{varid} - {fuzid}'] = truth_degree

        truth_degrees = {k:v for k,v in sorted(list(truth_degrees.items()), key=lambda x: x[1], reverse=True)}
        print('Степени истинности всех термов лингвистических переменных:')
        for k,v in truth_degrees.items(): print(f'«{k}» – {v}')


    def aggregate(self):
        """На этапе агрегирования необходимо вывести степени истинности антецедентов всех нечётких продукционных правил, представленных в системе (в убывающем порядке степени истинности)."""
        truth_degrees = {}
        for ruleid, rule in self.rules.items():
            truth_degrees[ruleid] = {}
            for antid, ant in rule['если'].items():
                truth_degrees[ruleid][f'{antid} - {ant}'] = self.variables[antid]['нечёткие переменные'][ant]['значение']
            value = get_output(rule, self.variables)
            truth_degrees[ruleid]['значение'] = value
            self.rules[ruleid]['значение'] = value
        truth_degrees = {k:v for k,v in sorted(list(truth_degrees.items()), key=lambda x: x[1]['значение'], reverse=True)}
        print('Cтепени истинности антецедентов всех нечётких продукционных правил:')
        for k, v in truth_degrees.items():
            print(f'«{k}» = {"min" if self.rules[ruleid]["операция"] == "AND" else "max"}({[vv for kk,vv in truth_degrees[k].items() if kk not in ("значение", "следствие")]}) = {v["значение"]}')

    def activate(self):
        """На этапе активизации необходимо построить графики функций принадлежности, составляющих консеквенты ядер всех нечетких продукционных правил. В алгоритме Мамдани используется min-активизация."""
        for ruleid, rule in self.rules.items():
            value = rule['значение']

            for consid, cons in rule['то'].items():
                if 'следствие' not in self.variables[consid]['нечёткие переменные'][cons]:
                    self.variables[consid]['нечёткие переменные'][cons]["следствие"] = [apply_min(self.variables[consid]['нечёткие переменные'][cons], value)]
                else: self.variables[consid]['нечёткие переменные'][cons]["следствие"].append(apply_min(self.variables[consid]['нечёткие переменные'][cons], value))

                graph_ant = clean(self.variables[consid]['нечёткие переменные'][cons])
                graph_cons = clean(self.variables[consid]['нечёткие переменные'][cons]["следствие"][-1])
                plt.plot(list(graph_ant.keys()), list(graph_ant.values()), label = 'до активизации')
                plt.plot(list(graph_cons.keys()), list(graph_cons.values()), label = 'после активизации')
                plt.title(f"{ruleid}: {consid} '{cons}'")
                plt.xlabel('y')
                plt.ylabel('μ(y)')
                plt.legend()
                plt.show()

    def accumulate(self):
        """На этапе аккумулирования заключений необходимо построить график функции принадлежности, полученной в результате аккумулирования заключений."""
        for varid, var in self.variables.items():
            graph = {}
            for fuzid, fuz in var['нечёткие переменные'].items():
                if 'следствие' in fuz:
                    for cons in fuz['следствие']:
                        if len(graph) == 0:
                            graph = cons
                        else: graph = update_graph(graph, cons)
                        # for x, y in cons.items():
                        #     if x not in graph: graph[x] = y
                        #     else: graph[x] = max(graph[x], y)
                if len(graph) > 0:
                    graph = {k:v for k,v in sorted(list(graph.items()), key = lambda x: x[0])}
                    self.variables[varid]['следствие'] = graph


        for varid, var in self.variables.items():
            if 'следствие' in var:
                for fuzid, fuz in var['нечёткие переменные'].items():
                    graph = clean(fuz)
                    plt.plot(graph.keys(), graph.values(), label=f'{fuzid}')

                plt.plot(list(var['следствие'].keys()), list(var['следствие'].values()), label='после аккумулирования заключений')
                plt.legend()
                plt.xlabel('y')
                plt.ylabel('μ(y)')
                plt.title(f'{varid}')
                plt.show()

    def defuzzificate(self):
        for varid, var in self.variables.items():
            if 'следствие' in var:
                graph = list(clean(var['следствие']).items())
                areas = []
                centroids = []
                ax = []
                for i in range(len(graph)-1):
                    x1,x2 = graph[i][0], graph[i+1][0]
                    y1,y2 = graph[i][1], graph[i+1][1]
                    area = (x2 - x1) * ((y1 + y2) / 2)
                    areas.append(area)

                    if y1 == 0: centroid = (x1 + x2 + x2) / 3
                    elif y2 ==0: centroid = (x1 + x1 + x2) / 3
                    elif y1 == y2: centroid = (x1 + x2) / 2
                    else: centroid = ((y2 + 2*y1) / (3*(y2 + y1))) * (x2 - x1)
                    centroids.append(centroid)

                    ax.append(area*centroid)
                value = sum(ax) / sum(areas) if sum(areas) != 0 else 0
                plt.plot([i[0] for i in graph], [i[1] for i in graph])
                plt.axvline(x = value)
                plt.xlabel('y')
                plt.ylabel('μ(y)')
                plt.title(f'{varid} = {value}')
                plt.show()


    def run(self):
        self.fuzzificate()
        self.aggregate()
        self.activate()
        self.accumulate()
        self.defuzzificate()


if __name__ == '__main__':
    input_vars = {
    'вес': 0.7,
    'ширина': 13.5,
    'длина': 27,
}

    f = Fuzzy(rules = yaml_load('продукционные правила.yaml'),
                objects = yaml_load('объекты.yaml'),
                variables = yaml_load('лингвистические переменные.yaml'),
                input_vars = input_vars)

    f.fuzzificate()
    f.aggregate()
    f.activate()
    f.accumulate()
    f.defuzzificate()
    