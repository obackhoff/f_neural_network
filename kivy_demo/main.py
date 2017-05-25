#!/usr/bin/env python
import matplotlib
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
from fnn import run
import sys
import matplotlib.pyplot as plt
import numpy as np
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvas

from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.app import App
from kivy.graphics import Color, Rectangle
from random import random as r
from functools import partial
from threading import Thread


np.set_printoptions(threshold=np.nan)


class nnDemoApp(App):
    icon = 'icon.png'
    title = 'FF Neural Network Demo'
    DATA_POINTS = 50
    NOISE_LEVEL = 3
    FUNC = 1
    EPOCHS = 40000
    LEARNING_RATE = 0.001
    LAMBDA = 0.1
    ACTIVATION = 'elu'
    OUT_ACTIVATION ='lin'
    SHAPE = 'Hidden Layer Nodes'

    fig, ax = plt.subplots()

    label_points = Label(text='Data points: ' + str(50))
    label_noise = Label(text='Noise level: ' + str(3))
    label_epochs = Label(text='Train epochs: ' + str(40000))
    label_learning = Label(text='Learning rate: ' + str(LEARNING_RATE))
    label_lambda = Label(text='Reg. factor: ' + str(LAMBDA))

    nn_shape = TextInput(text='Hidden Layer Nodes')
    output = TextInput(size_hint = (1, 0.5))

    spinner_func = Spinner(text='Polynomial',values=('Trigonometric', 'Polynomial', 'Line', 'Exponential 1', 'Exponential 2'))

    def OnSliderValueChange(self, instance, value):
        if instance.id == 'points':
            self.label_points.text = 'Data points: ' + str(value)
            self.DATA_POINTS = value
        elif instance.id == 'noise':
            self.label_noise.text = 'Noise level: ' + str('%.1f' % value)
            self.NOISE_LEVEL = value
        elif instance.id == 'epochs':
            self.label_epochs.text = 'Train epochs: ' + str(value)
            self.EPOCHS = value
        elif instance.id == 'learning_rate':
            self.label_learning.text = 'Learning rate: ' + str('%.5f' % value)
            self.LEARNING_RATE = value
        elif instance.id == 'lambda':
            self.label_lambda.text = 'Reg. factor: ' + str('%.5f' % value)
            self.LAMBDA = value

    def func(self, x):
        if(self.FUNC == 0):
            return 5*np.sin(x) + 5*np.cos(2*x)
        elif (self.FUNC == 1):
            return 0.1*x**4 - 2*x**2 - 2
        elif (self.FUNC == 2):
            return 0.5*x + 2
        elif (self.FUNC == 3):
            return 5*np.exp(-0.5*x**2)
        elif (self.FUNC == 4):
            return 5*x/(np.abs(x)+1.15)

    def set_func(self, spinner, function):
        if function == 'Trigonometric':
            self.FUNC = 0
            self.draw_base_func()
        elif function == 'Polynomial':
            self.FUNC = 1
            self.draw_base_func()
        elif function == 'Line':
            self.FUNC = 2
            self.draw_base_func()
        elif function == 'Exponential 1':
            self.FUNC = 3
            self.draw_base_func()
        elif function == 'Exponential 2':
            self.FUNC = 4
            self.draw_base_func()

    def set_activation(self, spinner, function):
        if function == 'ELU':
            if spinner.id == 'activ':
                self.ACTIVATION = 'elu'
            else:
                self.OUT_ACTIVATION = 'elu'
        elif function == 'Linear':
            if spinner.id == 'activ':
                self.ACTIVATION = 'lin'
            else:
                self.OUT_ACTIVATION = 'lin'
        elif function == 'Sigmoid':
            if spinner.id == 'activ':
                self.ACTIVATION = 'sigmoid'
            else:
                self.OUT_ACTIVATION = 'sigmoid'
        elif function == 'Softplus':
            if spinner.id == 'activ':
                self.ACTIVATION = 'softplus'
            else:
                self.OUT_ACTIVATION = 'softplus'
        elif function == 'Tanh':
            if spinner.id == 'activ':
                self.ACTIVATION = 'tanh'
            else:
                self.OUT_ACTIVATION = 'tanh'

    def draw_base_func(self):
        self.ax.clear()
        p = np.reshape(np.arange(-5,5,0.01),(-1,1))
        self.ax.plot(p, self.func(p), color='green', label='Truth', alpha=0.5)
        self.fig.canvas.draw()

    def async_train(self, fig, ax, txtfield, *largs):
        Thread(target=partial(self.train_nn, fig, ax, txtfield)).start()

    def train_nn(self, fig, ax, txtfield, *largs):
        x = 10 * np.random.random(self.DATA_POINTS) - 5
        #x += (self.NOISE_LEVEL*np.random.random(self.DATA_POINTS) - self.NOISE_LEVEL/2)
        y = self.func(x)
        y += (self.NOISE_LEVEL*np.random.random(self.DATA_POINTS) - self.NOISE_LEVEL/2)

        t = np.dstack((x,y))[0]
        t_str = np.array2string(t, separator=',').replace('\n', '')

        p = np.reshape(np.arange(-5,5,0.01),(-1,1))
        p_str = np.array2string(p, separator=',').replace('\n', '')

        self.SHAPE = self.nn_shape.text
        if self.SHAPE == 'Hidden Layer Nodes':
            self.SHAPE = '[1,8,8,1]'
            self.nn_shape.text = '8,8'
        else:
            if not self.SHAPE == '':
                self.SHAPE = '[1,'+ self.SHAPE+',1]'
            else:
                self.SHAPE = '[1,1]'

        sys.argv = ['prog','-Q', '-a', self.ACTIVATION,'-o', self.OUT_ACTIVATION,'-n', self.SHAPE, '-t', t_str, '-p', p_str, '-e', self.EPOCHS,'-r', 2000 , '-l', self.LEARNING_RATE, '-L', self.LAMBDA]
        preds = run(txtfield)

        ax.clear()
        ax.plot(p,preds, color='red', label='Prediction')
        ax.plot(p, self.func(p), color='green', label='Truth', alpha=0.5)
        ax.scatter(x, y, label='Training set')
        ax.legend()
        fig.canvas.draw()

        RMSE = np.sqrt((preds - self.func(p))**2 )
        self.output.text += '\nRMSE = ' + str(RMSE.sum()/len(RMSE))


    def build(self):

        canvas = self.fig.canvas

        ## main and data settings layout
        btn_run = Button(text='Run', on_press=partial(self.async_train, self.fig, self.ax, self.output))

        spinner_func = Spinner(text='Polynomial',values=('Trigonometric', 'Polynomial', 'Line', 'Exponential 1', 'Exponential 2'))
        self.spinner_func.bind(text=self.set_func)
        self.draw_base_func()

        slider_layout_points = BoxLayout(orientation='vertical')
        slider_points = Slider(min=5, max=200, value=50, step=1, value_track=True, id='points')
        slider_points.bind(value=self.OnSliderValueChange)

        slider_layout_points.add_widget(slider_points)
        slider_layout_points.add_widget(self.label_points)

        slider_layout_noise = BoxLayout(orientation='vertical')
        slider_noise = Slider(min=0, max=10, value=3, step=0.1, value_track=True, id='noise')
        slider_noise.bind(value=self.OnSliderValueChange)

        slider_layout_noise.add_widget(slider_noise)
        slider_layout_noise.add_widget(self.label_noise)

        self.output.readonly = True
        self.output.text += "Press 'Run' to train the Neural Network\nLOG:\n"


        ## Neural network settings layout

        spinner_activation = Spinner(text='ELU',values=('ELU','Sigmoid', 'Softplus', 'Tanh'), id='activ')
        spinner_out_activation = Spinner(text='Linear',values=('ELU','Linear','Sigmoid', 'Softplus', 'Tanh'), id='out_activ')
        spinner_activation.bind(text=self.set_activation)
        spinner_out_activation.bind(text=self.set_activation)

        slider_layout_epochs = BoxLayout(orientation='vertical')
        slider_epochs = Slider(min=1000, max=100000, value=self.EPOCHS, step=200, value_track=True, id='epochs')
        slider_epochs.bind(value=self.OnSliderValueChange)
        slider_layout_epochs.add_widget(slider_epochs)
        slider_layout_epochs.add_widget(self.label_epochs)

        slider_layout_learning = BoxLayout(orientation='vertical')
        slider_learning = Slider(min=0.0001, max=0.01, value=self.LEARNING_RATE, step=0.0001, value_track=True, id='learning_rate')
        slider_learning.bind(value=self.OnSliderValueChange)
        slider_layout_learning.add_widget(slider_learning)
        slider_layout_learning.add_widget(self.label_learning)

        slider_layout_lambda = BoxLayout(orientation='vertical')
        slider_lambda = Slider(min=0.0, max=0.5, value=self.LAMBDA, step=0.0001, value_track=True, id='lambda')
        slider_lambda.bind(value=self.OnSliderValueChange)
        slider_layout_lambda.add_widget(slider_lambda)
        slider_layout_lambda.add_widget(self.label_lambda)



        ## all layouts added to root lyout

        layout = BoxLayout(size_hint=(1, None), height=50)
        layout.add_widget(self.nn_shape)
        layout.add_widget(self.spinner_func)
        layout.add_widget(slider_layout_points)
        layout.add_widget(slider_layout_noise)
        layout.add_widget(btn_run)

        layout_nn = BoxLayout(size_hint=(1, None), height=50)

        layout_nn.add_widget(spinner_activation)
        layout_nn.add_widget(spinner_out_activation)
        layout_nn.add_widget(slider_layout_epochs)
        layout_nn.add_widget(slider_layout_learning)
        layout_nn.add_widget(slider_layout_lambda)

        root = BoxLayout(orientation='vertical')
        root.add_widget(canvas)
        root.add_widget(layout)
        root.add_widget(layout_nn)
        root.add_widget(self.output)

        return root


if __name__ == '__main__':
    nnDemoApp().run()
