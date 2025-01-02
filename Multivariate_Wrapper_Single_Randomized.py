"""
Multivariate
Wrapper	-> NB
Single	-> Accuracy
Randomized	->	Genetic Algo
"""
from utility import *

# Wrapper
nb = GaussianNB()

class FeatureSelection(BinaryProblem):

    def __init__(self, x: pd.DataFrame, y: pd.Series, pop_size: int, classifier):

        super(FeatureSelection, self).__init__()
        self.F = x
        self.target = y
        self.number_of_bits = x.shape[1]
        self.obj_labels = ["accuracy"]
        self.count = 0
        self.pop_size = pop_size
        self.classifier = classifier

        print(self.F.shape)

    def name(self):
        return "My Feature Selection Problem"

    def number_of_variables(self):
        return 1

    def number_of_objectives(self):
        return 1

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

        X_train, X_test, y_train, y_test = train_test_split(self.F[idx], self.target, test_size=0.30, random_state=42)

        X_res, y_res = smote.fit_resample(X_train, y_train)

        nb.fit(X_res, y_res)
        y_pred = nb.predict(X_test)

        # model accuracy
        accuracy = accuracy_score(y_test, y_pred)

        solution.objectives[0] = accuracy * -1.0

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

X_wrapper = X
attribute_names = X_wrapper.columns

population_size = 22
max_evaluations = population_size*50

problem = FeatureSelection(X_wrapper, y, population_size, nb)
print(problem.number_of_objectives)

algorithm=GeneticAlgorithm(
    problem=problem,
    population_size=population_size,
    offspring_population_size=population_size,
    mutation=BitFlipMutation(probability=(1 / X_wrapper.shape[1])),
    crossover=SPXCrossover(probability=1),
    termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
)

print("Initial population")
start = time()
algorithm.run()
wrapper_time = time()-start
solutions = algorithm.get_result()
print(solutions)

print("Final population")
selected = [attribute_names[j] for j in range(len(solutions.get_binary_string())) \
            if solutions.get_binary_string()[j] == "1"]
selected = "; ".join(selected)

acc = solutions.objectives[0]

print(f"Time: {wrapper_time}")
print(f"Subset: {selected}")
