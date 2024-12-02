from blahutArimotoTorch import BlahutArimoto
from tqdm import tqdm  # for progress bar
from datetime import datetime
import plotly.graph_objects as go

class VisualizeBA():
    def __init__(self) -> None:
        self.records = []
        self.As = []
        self.capacities = []
        self.theoretical_capacitys = []

    def addData(self, data):
        self.As.append(data["A"])
        self.capacities.append(data["capacity"])
        self.records.append(data)
        self.theoretical_capacitys.append(data["theoretical_c"])

    def clearData(self):
        self.records = []
        self.As = []
        self.capacities = []
        self.theoretical_capacitys = []

    def plot_A_vs_capacity(self):
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x = self.As,
            y = self.capacities,
            mode='lines',
            name='Channel Capacity'
        ))

        fig.update_layout(
            title="Channel Capacity vs. Peak Power Constraint",
            xaxis_title="A (Peak Power Constraint)",
            yaxis_title="Channel Capacity (bits)",
            template="plotly_dark",
            showlegend=False,
            autosize=True,
        )

        fig.show()

        # Print results
        for i in range(len(self.As)):
            print(f"A = {self.As[i]:.2f} | Channel Capacity = {self.capacities[i]:.3f}")


    def plot_capcity_vs_theoretical_capacity(self):
        fig = go.Figure()

        # Add the computed capacity
        fig.add_trace(go.Scatter(
            x=self.As,
            y=self.capacities,
            mode='lines',
            name='Computed Capacity'
        ))

        # Add the theoretical capacity
        fig.add_trace(go.Scatter(
            x=self.As,
            y=self.theoretical_capacitys,
            mode='lines',
            name='Theoretical Capacity'
        ))

        fig.update_layout(
            title="Channel Capacity vs. Peak Power Constraint",
            xaxis_title="A (Peak Power Constraint)",
            yaxis_title="Capacity (bits)",
            template="plotly_dark",
            showlegend=True,
            autosize=True,
        )

        fig.show()

        # Print results
        for i in range(len(self.As)):
            print(f"A = {self.As[i]:.2f} | Channel Capacity = {self.capacities[i]:.3f} Theoretical = {self.theoretical_capacitys[i]:.3f}")

    def plot_A_vs_p_x(self):
        outputs = self.records
        fig = go.Figure()

        for i, output in enumerate(outputs):
            fig.add_trace(go.Scatter(
                x=output["x"],    # x-axis: the input symbols (discretized x)
                y=output["p_x"],  # y-axis: the probability distribution p(x)
                mode='lines',
                name=f"A = {(output['A']):.2f}",
            ))

        fig.update_layout(
            title="Input Probability Distributions for Different A",
            xaxis_title="x (Input Symbol)",
            yaxis_title="p(x) (Probability Distribution)",
            template="plotly_dark",
            showlegend=True,
            autosize=True,
        )

        fig.show()


    def plot_A_vs_p_x_dynamic(self):
        outputs = self.records
        fig = go.Figure()

        # Add traces for each A, initially all hidden
        for i, output in enumerate(outputs):
            fig.add_trace(go.Scatter(
                x=output["x"],    # x-axis: the input symbols (discretized x)
                y=output["p_x"],  # y-axis: the probability distribution p(x)
                mode='lines',
                name=f"A = {output['A']:.2f}",
                visible='legendonly'  # Initially, all traces are hidden
            ))

        # Create the slider steps to control the visibility of traces
        steps = []
        for i, output in enumerate(outputs):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(outputs)},  # Hide all traces
                      {"title": f"A = {output['A']:.2f}"}],  # Set the title to the selected A value
            )
            step["args"][0]["visible"][i] = True  # Make the current trace visible
            steps.append(step)

        # Add the slider to control A
        fig.update_layout(
            title="Input Probability Distributions for Different A",
            xaxis_title="x (Input Symbol)",
            yaxis_title="p(x) (Probability Distribution)",
            template="plotly_dark",
            showlegend=True,
            sliders=[dict(
                currentvalue={"prefix": f"A = ", "visible": True, "xanchor": "center"},
                steps=steps
            )]
        )

        # Show the dynamic plot
        fig.show()

        return fig


    def plot_A_vs_p_y(self):
        fig = go.Figure()

        for i, output in enumerate(self.records):
            fig.add_trace(go.Scatter(
                x=output["y"],  # x-axis: the output symbols (discretized y)
                y=output["p_y"],  # y-axis: the probability distribution p(y)
                mode='lines',
                name=f"A = {self.As[i]:.2f}",
            ))

        fig.update_layout(
            title="Output Probability Distributions for Different A",
            xaxis_title="y (Output Symbol)",
            yaxis_title="p(y) (Probability Distribution)",
            template="plotly_dark",
            showlegend=True,
            autosize=True,
        )

        fig.show()


    def plot_p_x_records(self, idx=-1):
        """   !!!!STATIC!!!
        Example usage: Assuming p_x_records is a list of probability distributions for each iteration
        p_x_records = [p_x_iteration_1, p_x_iteration_2, ..., p_x_iteration_n]"""

        fig = go.Figure()

        output = self.records[idx]

        for i, p_x in enumerate(output["p_x_records"]):
            fig.add_trace(go.Scatter(
                x=output["x"],    # x-axis: the input symbols (discretized x)
                y=p_x,  # y-axis: the probability distribution p(x)
                mode='lines',
                name=f"Record {(i+1)}",
            ))

        fig.update_layout(
            title="Input Probability Distributions Over Iterations",
            xaxis_title="x (Input Symbol)",
            yaxis_title="p(x) (Probability Distribution)",
            template="plotly_dark",
            showlegend=True,
            autosize=True,
        )

        fig.show()

    def plot_p_x_records_dynamic(self, idx=-1):
        """Example usage: Assuming p_x_records is a list of probability distributions for each iteration
        p_x_records = [p_x_iteration_1, p_x_iteration_2, ..., p_x_iteration_n]"""

        # Create an empty figure
        fig = go.Figure()

        output = self.records[idx]

        # Add traces for each iteration (initially all traces hidden)
        for i, p_x in enumerate(output["p_x_records"]):
            fig.add_trace(go.Scatter(
                x=output["x"],  # x-axis: the input symbols (discretized x)
                y=p_x,  # y-axis: the probability distribution p(x)
                mode='lines',  # Connect the points with lines
                name=f"A = {output['A']:.2f}, Record {(i+1)}",  # Label for the iteration
                visible='legendonly',  # Initially set to not visible
            ))

        # Create the slider steps (one per iteration)
        steps = []
        for i in range(len(output["p_x_records"])):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(output["p_x_records"])},  # Hide all traces
                    {"title": f"Record {(i+1)}"}],  # Update the title
            )
            step["args"][0]["visible"][i] = True  # Make the current iteration visible
            steps.append(step)

        # Add the slider to the layout
        fig.update_layout(
            title="Input Probability Distributions Over Iterations",
            xaxis_title="x (Input Symbol)",
            yaxis_title=f"p(x) (Probability Distribution)",
            template="plotly_dark",  # Optional: use a dark theme
            showlegend=True,
            sliders=[dict(
                currentvalue={"prefix": f"Iteration: ", "visible": True, "xanchor": "center"},
                steps=steps
            )]
        )

        # Show the dynamic plot
        fig.show()

        # print
        print("==========")
        print(f"Total Iterations: {(output['iter'])}")
        print(f"A = {output['A']:.2f}")
        print(f"Channel Capacity = {output['capacity']:.3f}")
        print(f"Theoretical Capacity = {output['theoretical_c']:.3f}")
        print(f"Mean power = {output['mean_power']:.3f}")


        def savePlot(self, fig, filename, ext="html"):
            # Save the interactive graph as an HTML file
            fig.write_html(f"{filename}.{ext}")


class Experiment(BlahutArimoto, VisualizeBA):
    def __init__(self, NX=500, NY=1000, sigma=1, max_iter=10000, tolerance=1e-6, epsilon=1e-12,
                 printInit=False, earlyStop=True, device=None):
        # Initialize both parents' constructors
        BlahutArimoto.__init__(self, sigma=sigma, max_iter=max_iter, NX=NX, NY=NY, tolerance=tolerance, 
                               epsilon=epsilon, printInit=printInit, earlyStop=earlyStop, device=device)
        print("BlahutArimoto initialized")

        VisualizeBA.__init__(self)
        print("Visualizer initialized")

        print("Experiment initialized")  

    def run(self, As):
        self.clearData()
        print("================")
        print(f"Experiment started for {len(As)} many As")
        print(f"device {self.device}")
        print(f"NX = {self.NX}, NY = {self.NY}")
        print(f"Tolerance = {self.tolerance}, epsilon = {self.epsilon}")
        print(f"Max Iterations = {self.max_iter}")
        print("================")
        with torch.no_grad():
            for Ai in As:
                self.A = Ai
                result = self.runAlgorithm()
                self.addData(result)
        print("================")
        print("Experiment finished")
        print("================")


    def appendRunHistory(self, anotherExperiment: "Experiment"):
        if not isinstance(anotherExperiment, Experiment):
            raise TypeError(f"Expected an instance of 'Experiment', but got {type(anotherExperiment).__name__}.")
        
        self.records.extend(anotherExperiment.records)
        self.As.extend(anotherExperiment.As)
        self.capacities.extend(anotherExperiment.capacities)


