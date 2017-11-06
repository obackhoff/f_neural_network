 # FF Neural Network GUI Demo

 ![Screenshot](https://obackhoff.github.io/data/ffdemo.png)


 This is a GUI application which demonstrates how neural networks can be used for regression tasks. Different settings and topologies can be applied to the neural network in order to see their effect on the outcome.

 The default options: "ELU" and "Linear" are the activation functions in the hidden layers and the output layer, respectively.

 To run it, you will need a couple of dependencies:

 ``` bash
 sudo pip install kivy kivy-garden numpy matplotlib
 garden install matplotlib
 ```

 The kivy-garden.matplotlib is required so that matplotlib can be used within a Kivy application.

 Then simply double-click main.py if set to executable or run:

 ``` bash
 python main.py
 ```