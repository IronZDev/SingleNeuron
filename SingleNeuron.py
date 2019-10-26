import random
import PySimpleGUI as sg
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Neuron():

    def __init__(self, size, activation='heaviside', learning_rate=0.01, boundary=0):
        # seeding for random number generation
        np.random.seed(1)
        self.weights = np.random.uniform(0, 1, 3)
        self.learning_rate = learning_rate
        self.boundary = boundary
        if activation is 'heaviside':
            self.activation_function = self.heaviside
        elif activation is 'sigmoid':
            self.activation_function = self.sigmoid

    def sigmoid(self, x, derivative=False):
        if not derivative:
            return 1 / (1 + np.exp(-x))
        else:
            return self.sigmoid(x) * (1 - self.sigmoid(x))

    def heaviside(self, x, derivative=False):
        if not derivative:
            return np.heaviside(x, 1)
        else:  # Derivative mode
            return 1

    def train(self, training_inputs, training_outputs, training_iterations):
        # training the model to make accurate predictions while adjusting weights continually
        for _ in range(training_iterations):
            for input_val, output in zip(training_inputs, training_outputs):
                calculated = self.predict(self.weights @ input_val)
                error = output - calculated
                adjustments = self.learning_rate * error * self.activation_function(self.weights @ input_val, True) * input_val
                self.weights += adjustments
                # print(self.weights)
            # # siphon the training data via  the neuron
            # output = self.learn(training_inputs)
            # # print(output)
            # # print(training_outputs)
            # # computing error rate
            # error = training_outputs - output
            # print(error)
            # # performing weight adjustments
            # print(training_inputs)
            # print(self.sigmoid_derivative(output))
            # adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            # print(adjustments)
            # print(self.weights)
            # self.weights += adjustments
            # print(self.weights)

    def predict(self, ins):
        # passing the inputs via the neuron to get output
        ins = ins.astype(float)
        # output = self.activation_function(np.dot(ins, self.weights[1:]))
        output = self.activation_function(ins)
        return output


# Calculating gaussian distribution
def calculate_gaussian(size=1000, number_of_clusters=3, mean_min=-10.0, mean_max=10.0, std_deviation_min=0.5,
                       std_deviation_max=2.0):
    # Generate random clasters in 2d
    colors_available = ['r', 'g']
    data = [[], [], []]
    for cluster in range(number_of_clusters):
        center = [random.uniform(mean_min, mean_max), random.uniform(mean_min, mean_max)]
        x, y = np.random.multivariate_normal(center,
                                             [[random.uniform(std_deviation_min, std_deviation_max), 0],
                                              [0, random.uniform(std_deviation_min, std_deviation_max)]],
                                             size).T
        data[0].append(x)
        data[1].append(y)
        # Assign a color to each cluster
        data[2].extend([colors_available[cluster % 2] for i in range(size)])
    # Merge points together
    data[0] = np.concatenate((data[0][0], data[0][1]))
    data[1] = np.concatenate((data[1][0], data[1][1]))
    return data


# Generating plot
def generate_plot(x):
    plt.scatter(x[0], x[1], c=x[2])
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    return plt.gcf()


# Helper functions
# Draw plot on canvas
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


# Validate data
def is_valid():
    for i, val in enumerate(values):
        if (type(values[val]) is str or type(values[val]) is int) and not val == 'activation':
            # Check if values are not empty
            if len(str(values[val])) == 0:
                window[val].Update('')
                sg.popup_error('Invalid values!', 'Value of '+column1[i-1][0].DisplayText+' is invalid')
                return False
            # Check if values are numbers
            if not is_number(values[val]):
                window[val].Update('')
                sg.popup_error('Invalid values!', 'Value of '+column1[i-1][0].DisplayText+' is invalid')
                return False
            # Check deviations
            if 'Deviation' in val and float(values[val]) < 0:
                window[val].Update('')
                sg.popup_error('Invalid values!', 'Value of '+column1[i-1][0].DisplayText+' can\'t be lower than 0')
                return False
    if float(values['minDeviation']) > float(values['maxDeviation']):
        sg.popup_error('Invalid values!', 'Min std deviation can\'t be greater than max std deviation')
        return False
    # Check number of clusters
    if int(values['clustersNum']) <= 0 or int(values['clustersNum']) > 7:
        sg.popup_error('Invalid values!', 'Clusters number can\'t be lower than 1 and greater than 7')
        return False
    # Check size
    if int(values['clustersNum']) > int(values['size']):
        sg.popup_error('Invalid values!', 'Size can\'t be lower than number of clusters')
        return False
    # Check learning rate
    if float(values['learningRate']) <= 0 or float(values['learningRate']) > 1:
        sg.popup_error('Invalid values!', 'Learning rate can\'t be lower/equal to 0 and greater than 1')
        return False
    return True


# GUI
column1 = [
    [sg.Text('Min mean value')],
    [sg.Text('Max mean value')],
    [sg.Text('Min std deviation')],
    [sg.Text('Max std deviation')],
    [sg.Text('Number of values to generate per cluster')],
    [sg.Text('Number of clusters')],
    [sg.Text('Activation function')],
    [sg.Text('Number of epochs')],
    [sg.Text('Learning rate')],
    [sg.Button('Draw', size=(15, 1))]
]

column2 = [
    [sg.In(default_text='-10.0', key='minVal', size=(4, 1))],
    [sg.In(default_text='10.0', key='maxVal', size=(4, 1))],
    [sg.In(default_text='0.5', key='minDeviation', size=(4, 1))],
    [sg.In(default_text='2.0', key='maxDeviation', size=(4, 1))],
    [sg.Spin([i for i in range(100, 100000, 100)], initial_value=100, key='size', size=(5, 1))],
    [sg.Spin([i for i in range(1, 8)], initial_value=2, key='clustersNum', size=(1, 1))],
    [sg.InputCombo(('heaviside', 'sigmoid'), default_value='heaviside', size=(10, 1), key='activation')],
    [sg.Spin([i for i in range(10, 100000, 10)], initial_value=100, key='epochs', size=(5, 1))],
    [sg.In(default_text='0.1', key='learningRate', size=(4, 1))],
    [sg.Exit(size=(8, 1))]
]

layout = [
    [sg.Canvas(size=(1, 1), key='canvas')],
    [sg.Column(column1), sg.Column(column2)]
]

# Create the Window
window = sg.Window('Single neuron', layout).Finalize()

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event in (None, 'Exit'):  # if user closes window or clicks cancel
        print('Close')
        plt.close('all')
        break
    if event == 'Draw':
        if is_valid():
            points = calculate_gaussian(int(values['size']), int(values['clustersNum']), float(values['minVal']),
                                      float(values['maxVal']), float(values['minDeviation']),
                                      float(values['maxDeviation']))
            inputs = []
            outputs = []
            for i in range(len(points[0])):
                inputs.append([1.0, points[0][i], points[1][i]])
                if points[2][i] is 'r':
                    outputs.append(0)
                else:
                    outputs.append(1)
            print(inputs)
            neuron = Neuron(int(values['size']), values['activation'], float(values['learningRate']))
            # print(values['activation'])
            neuron.train(np.asarray(inputs), np.asarray(outputs), int(values['epochs']))
            decision_x = np.linspace(np.amin(points[0]), np.amax(points[0]))
            decision_y = -(neuron.weights[0] + neuron.weights[1] * decision_x)/neuron.weights[2]
            # for i in decision_x:
            #     # y=ax+b calculated for both min and max points
            #     decision_y.append((-neuron.weights[1]/neuron.weights[2])*i-neuron.weights[0]/neuron.weights[2])

            if 'fig_canvas_agg' in globals():  # Update if plot already exists
                plt.clf()
                plt.scatter(points[0], points[1], c=points[2])
                plt.plot(decision_x, decision_y, linestyle='-')
                fig_canvas_agg.draw()
            else:  # Generate new plot
                fig = generate_plot(points)
                plt.plot(decision_x, decision_y, linestyle='-')
                fig_canvas_agg = draw_figure(window['canvas'].TKCanvas, fig)

window.close()
