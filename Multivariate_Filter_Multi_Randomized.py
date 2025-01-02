"""
Multivariate
Filter	->  mRMR, CMIM
Multi	-> Relevance/Redundancy, Mutual Information
Randomized	->	NSGA-II
"""
from utility import *
# Wrapper
class FeatureSelection(BinaryProblem):

    def __init__(self, x: pd.DataFrame, y: pd.Series, pop_size: int, classifier):

        super(FeatureSelection, self).__init__()
        self.F = x
        self.target = y
        self.number_of_bits = x.shape[1]
        self.obj_labels = ["mRMR", "CMIM"]
        self.count = 0
        self.pop_size = pop_size
        self.classifier = classifier

        print(self.F.shape)

    def name(self):
        return "My Feature Selection Problem"

    def number_of_variables(self):
        return 1

    def number_of_objectives(self):
        return 2

    def number_of_constraints(self):
        return 1

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        idx = []

        self.__evaluate_constraints(solution)
        # check constraint
        if solution.constraints[0] == 0:
            check = random.randrange(self.number_of_bits)
            solution.variables[0] = [True if _ == check else False for _ in range(self.number_of_bits)]

        for index, bits in enumerate(solution.variables[0]):
            if bits:
                idx.append(self.F.columns[index])

        x_train = self.F[idx]

        # mrmr
        idx, mrmr_score, _ = MRMR.mrmr(x_train.values, self.target.values, n_selected_features=len(idx))
        # cmim
        idx, cmim_score, _ = CMIM.cmim(x_train.values, self.target.values, n_selected_features=len(idx))

        """
        mrmr_score = calculate_mrmr(x_train, y)
        cmim_score = calculate_cmim(x_train, y)
        """
        solution.objectives[0] = np.average(mrmr_score) * -1
        solution.objectives[1] = np.average(cmim_score) * -1

        if self.count < self.pop_size:
            print(*solution.objectives, solution.get_binary_string(), sep=",")
            self.count += 1

        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables(),
                                      number_of_objectives=self.number_of_objectives())
        new_solution.variables[0] = [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]

        return new_solution

    def __evaluate_constraints(self, solution: BinarySolution) -> None:
        solution.constraints = [0 for _ in range(self.number_of_constraints())]
        x = solution.variables
        if sum(x[0]) == 0:
            solution.constraints[0] = 0
        else:
            solution.constraints[0] = 1

    def get_name(self) -> str:
        return 'Feature selection'

X, y = smote.fit_resample(X, y)
attribute_names = X.columns

population_size = 42
max_evaluations = population_size*50

problem = FeatureSelection(X, y, population_size, nb)

algorithm=NSGAII(
    problem=problem,
    population_size=population_size,
    offspring_population_size=population_size,
    mutation=BitFlipMutation(probability=(1 / X.shape[1])),
    crossover=SPXCrossover(probability=1),termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
)

print("Initial population")
start = time()
algorithm.run()
wrapper_time = time()-start
solutions = get_non_dominated_solutions(algorithm.get_result())

print("Final population")
final_sorted = []
for i in range(len(solutions)):
    subset = []
    for index, bits in enumerate(list(map(int, solutions[i].get_binary_string()))):
        if bits:
            subset.append(X.columns[index])

    selected = [attribute_names[j] for j in range(len(solutions[i].get_binary_string())) \
                if solutions[i].get_binary_string()[j] == "1"]
    selected = "; ".join(selected)

    current_solution = solutions[i].objectives
    current_solution.append(selected)

    final_sorted.append(tuple(current_solution))

final_pop = sorted(final_sorted)
for ind in final_pop:
    print(*ind, sep=",")
print(f"Time: {wrapper_time}")
