import gradio as gr
import numpy as np
import torch
import re
import torch.nn as nn
import torch.optim as optim
from sympy import symbols, Eq, solve, sin, cos, tan, exp, log, E, sympify
from random import choice


class PolicyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RewardModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TreeNode:
    """Represents a node in the MCTS tree."""

    def __init__(self, state, parent=None):
        self.state = state  # Current state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.q_value = 0.0  # Accumulated rewards

    def is_fully_expanded(self):
        return len(self.children) > 0

    def best_child(self, exploration_weight=1.4):
        """Select the best child using UCT formula."""
        def uct_value(child):
            return (child.q_value / (child.visits + 1e-6)) + exploration_weight * np.sqrt(
                np.log(self.visits + 1) / (child.visits + 1e-6)
            )

        return max(self.children, key=uct_value)

    def add_child(self, child_state):
        """Add a child node with the given state."""
        child = TreeNode(state=child_state, parent=self)
        self.children.append(child)
        return child


class MathSolver:
    def __init__(self, dataset=None):
        self.dataset = dataset or []  # Dataset of math problems
        self.policy_model = PolicyModel(input_size=128, hidden_size=64, output_size=4)
        self.reward_model = RewardModel(input_size=128, hidden_size=64, output_size=1)
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=0.001)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=0.001)
        self.execution_context = {}

    def encode_problem(self, problem):
        """Advanced encoding using symbolic representation and problem length."""
        variables = len(re.findall(r'[a-zA-Z]', problem))
        operators = len(re.findall(r'[\+\-\*/\^]', problem))
        problem_length = len(problem)
        return np.array([variables, operators, problem_length] + [0] * 125)

    def policy_model_predict(self, equation1, equation2=None):
        """Predict steps to solve the equations."""
        try:
            equations = []
            if equation1:
                # Parse equation by splitting on '='
                lhs, rhs = equation1.strip().split('=')
                equations.append(Eq(sympify(lhs), sympify(rhs)))
            if equation2:
                lhs, rhs = equation2.strip().split('=')
                equations.append(Eq(sympify(lhs), sympify(rhs)))
            all_variables = set()
            for eq in equations:
                all_variables.update(eq.free_symbols)
            var_definitions = [f"{v} = symbols('{v}')" for v in all_variables]
            steps = [
                ("Define variables", "\n".join(var_definitions)),
                ("Define equation(s)", f"equations = {equations}"),
                ("Solve equation(s)", f"solution = solve(equations, {list(all_variables)})"),
                ("Print solution", "print(solution)"),
            ]
            return steps
        except Exception as e:
            print(f"Error during policy model prediction: {e}")
            return []

    def reward_model_predict(self, steps, success):
        """Predict reward based on steps and success."""
        encoded_steps = self.encode_problem(str(steps))
        encoded_steps = torch.tensor(encoded_steps, dtype=torch.float32)
        reward = self.reward_model(encoded_steps)
        return reward.item() if success else -reward.item()

    def execute_code(self, code):
        """Execute the generated code."""
        try:
            # Ensure necessary imports and variables are in the execution context
            exec("from sympy import symbols, Eq, solve, sin, cos, tan, exp, log, E", self.execution_context)
            # Dynamically initialize variables in the context
            for var_def in self.execution_context.get("var_definitions", []):
                exec(var_def, self.execution_context)
            exec(code, self.execution_context)
            return True
        except Exception as e:
            print(f"Error executing code: {e}")
            return False

    def mcts(self, equation1, equation2=None, num_rollouts=10):
        """Monte Carlo Tree Search for solving equations."""
        root = TreeNode(state=(equation1, equation2))
        for _ in range(num_rollouts):
            # Selection
            node = root
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
            # Expansion
            if not node.is_fully_expanded():
                steps = self.policy_model_predict(*node.state)
                for step, code in steps:
                    child_state = (step, code)
                    node.add_child(child_state)
            # Simulation
            success = True
            for step, code in steps:
                if not self.execute_code(code):
                    success = False
                    break
            # Backpropagation
            reward = self.reward_model_predict(steps, success)
            while node:
                node.visits += 1
                node.q_value += reward
                node = node.parent
        return root.best_child().state if root.children else None

    def solve(self, equation1, equation2=None):
        """Solve the given equations and return the steps."""
        self.execution_context = {}
        steps = self.policy_model_predict(equation1, equation2)
        variables = set()
        for eq in [equation1, equation2] if equation2 else [equation1]:
            if eq:
                lhs, rhs = eq.strip().split('=')
                variables.update(sympify(lhs).free_symbols)
                variables.update(sympify(rhs).free_symbols)
        self.execution_context["var_definitions"] = [f"{v} = symbols('{v}')" for v in variables]
        steps_output = ["Best solution found:"]
        for step, code in steps:
            steps_output.append(f"Step: {step}")
            steps_output.append(f"Code: {code}")
            if self.execute_code(code):
                steps_output.append("Execution successful.")
            else:
                steps_output.append("Execution failed.")
        if "solution" in self.execution_context:
            final_answer = self.execution_context["solution"]
            if isinstance(final_answer, dict):
                for var, value in final_answer.items():
                    steps_output.append(f"{var} = {value}")
            elif isinstance(final_answer, list):
                for solution in final_answer:
                    if isinstance(solution, tuple):
                        for idx, var in enumerate(variables):
                            steps_output.append(f"{list(variables)[idx]} = {solution[idx]}")
                    else:
                        steps_output.append(f"Solution: {solution}")
            else:
                steps_output.append(f"Final Answer: {final_answer}")
        else:
            steps_output.append("No final answer found.")
        return "\n".join(steps_output)


# Gradio Interface
def solve_math_problem(equation1, equation2=None):
    solver = MathSolver()
    return solver.solve(equation1, equation2)


with gr.Blocks() as app:
    gr.Markdown("# Math Problem Solver with Advanced Multi-Step Reasoning and Learning")
    with gr.Row():
        equation1_input = gr.Textbox(label="Enter the first equation (e.g., x + 5 = 10)", placeholder="x + 5 = 10")
        equation2_input = gr.Textbox(label="Enter the second equation (optional, e.g., x - y = 1)", placeholder="x - y = 1")
    solve_button = gr.Button("Solve")
    solution_output = gr.Textbox(label="Solution", interactive=False)
    solve_button.click(solve_math_problem, inputs=[equation1_input, equation2_input], outputs=[solution_output])

app.launch(debug=True)