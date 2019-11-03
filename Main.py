import PySimpleGUI as sg
from PlotHandler import *
from UtililityFunctions import *
from Neuron import Neuron

points = [[], [], []]


# Validate data
def is_valid():
    for i, val in enumerate(values):
        if (type(values[val]) is str or type(values[val]) is int) and not val == 'activation':
            # Check if values are not empty
            if len(str(values[val])) == 0:
                window[val].Update('')
                sg.popup_error('Invalid values!', 'Value of ' + column1[i - 1][0].DisplayText + ' is invalid')
                return False
            # Check if values are numbers
            if not is_number(values[val]):
                window[val].Update('')
                sg.popup_error('Invalid values!', 'Value of ' + column1[i - 1][0].DisplayText + ' is invalid')
                return False
            # Check deviations
            if 'Deviation' in val and float(values[val]) < 0:
                window[val].Update('')
                sg.popup_error('Invalid values!',
                               'Value of ' + column1[i - 1][0].DisplayText + ' can\'t be lower than 0')
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
    [sg.Button('Generate inputs', size=(15, 1)), sg.Button('Draw', size=(15, 1), disabled=True)]
]

column2 = [
    [sg.In(default_text='-10.0', key='minVal', size=(4, 1))],
    [sg.In(default_text='10.0', key='maxVal', size=(4, 1))],
    [sg.In(default_text='0.5', key='minDeviation', size=(4, 1))],
    [sg.In(default_text='2.0', key='maxDeviation', size=(4, 1))],
    [sg.Spin([i for i in range(100, 100000, 100)], initial_value=100, key='size', size=(5, 1))],
    [sg.Spin([i for i in range(1, 8)], initial_value=2, key='clustersNum', size=(1, 1))],
    [sg.InputCombo(('heaviside', 'sigmoid', 'sin', 'cos', 'tanh', 'sign', 'ReLu', 'leaky ReLu'),
                   default_value='heaviside', size=(10, 1), key='activation')],
    [sg.Spin([i for i in range(10, 100000, 10)], initial_value=1000, key='epochs', size=(5, 1))],
    [sg.In(default_text='0.01', key='learningRate', size=(4, 1))],
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

    if event == 'Generate inputs':
        if is_valid():
            points = calculate_gaussian(int(values['size']), int(values['clustersNum']), float(values['minVal']),
                                        float(values['maxVal']), float(values['minDeviation']),
                                        float(values['maxDeviation']))
            window.Element('Draw').Update(disabled=False)

    if event == 'Draw':
        if is_valid():
            inputs = []
            outputs = []
            for i in range(len(points[0])):
                inputs.append([1.0, points[0][i], points[1][i]])
                if points[2][i] is 'r':
                    outputs.append(0)
                else:
                    outputs.append(1)
            neuron = Neuron(values['activation'].lower().replace(' ', '_'),
                            float(values['learningRate']))
            neuron.train(np.asarray(inputs), np.asarray(outputs), int(values['epochs']))

            if 'fig_canvas_agg' in globals():  # Update if plot already exists
                plt.clf()
                plt.scatter(points[0], points[1], c=points[2])
                draw_decision_surface(neuron, np.amin(points[0]), np.amax(points[0]), np.amin(points[1]),
                                      np.amax(points[1]))
                fig_canvas_agg.draw()
            else:  # Generate new plot
                fig = generate_plot(points)
                draw_decision_surface(neuron, np.amin(points[0]), np.amax(points[0]), np.amin(points[1]),
                                      np.amax(points[1]))
                fig_canvas_agg = draw_figure(window['canvas'].TKCanvas, fig)

window.close()
