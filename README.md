## Feature maps produced for the digit 4 using LIF Neurons in the SNN:

<div>
<img src="https://github.com/jk-/snn-mnist/blob/master/plots/feature_map_0_4.png" width="200" align="left">
<img src="https://github.com/jk-/snn-mnist/blob/master/plots/feature_map_1_4.png" width="200" align="left">
<img src="https://github.com/jk-/snn-mnist/blob/master/plots/feature_map_2_4.png" width="200" align="left">
<img src="https://github.com/jk-/snn-mnist/blob/master/plots/feature_map_3_4.png" width="200" align="left">

<img src="https://github.com/jk-/snn-mnist/blob/master/plots/feature_map_4_4.png" width="200" align="left">
<img src="https://github.com/jk-/snn-mnist/blob/master/plots/feature_map_5_4.png" width="200" align="left">
<img src="https://github.com/jk-/snn-mnist/blob/master/plots/feature_map_6_4.png" width="200" align="left">
<img src="https://github.com/jk-/snn-mnist/blob/master/plots/feature_map_7_4.png" width="200" align="left">

<img src="https://github.com/jk-/snn-mnist/blob/master/plots/feature_map_8_4.png" width="200" align="left">
<img src="https://github.com/jk-/snn-mnist/blob/master/plots/feature_map_9_4.png" width="200" align="left">
<img src="https://github.com/jk-/snn-mnist/blob/master/plots/feature_map_10_4.png" width="200" align="left">
<img src="https://github.com/jk-/snn-mnist/blob/master/plots/feature_map_11_4.png" width="200" align="left">
</div>
<p></p>

https://www.researchgate.net/publication/324270622_Spiking_neural_networks_for_handwritten_digit_recognition-Supervised_learning_and_network_optimization?enrichId=rgreq-a04708b3adff7f03048e75efd7ad1ecf-XXX&enrichSource=Y292ZXJQYWdlOzMyNDI3MDYyMjtBUzo2MjQwMzE5Njc0ODU5NTRAMTUyNTc5MjIyMzYxNQ%3D%3D&el=1_x_3&_esc=publicationCoverPdf


*Goal: is to train a SNN on the mnist digit set*

Terms

El := resting potential
Vt := action potential

V(t) >= Vt a spike is issued
    and reset to El

Defaults:
    El = -70mV
    Vt = 20mV
    C = 300pF
    gl = 30nS

To enter refactor period after an action potential is triggered
you hold V(t) = El for a short period tref = 3ms and limit the
membrane potential to the range [El, Vt] through clipping

w := strength (weight) of a spike arriving at a synapse
will generate a post-synaptic current lsyn(t) in downstream neuron


lsyn(t) = w x c(t)
where C(t) = summation of chhange (t-t^i) * (e^-t/t^1 - e^-t/t^2)

t^1 := time of issue of thhe ith incoming spike
* := convolution operator
t1 = 5ms and t2 = 1.25ms model the c(t) and denote its falling and rising contstent

MNIST = 28x28

input layers of 784 (28*28)
hidden layer = 8112 neurons thhrough 12 a priori fixed 3x3 kernels
output layer has 10 neurons, each representing a digit (0-9)

NormAD algorithm

Because the mnist are static images we convert the images to grayscale
each pixel value in the range 0-255, to currents that can be applied as inputs to spiking neuron. Each pixel value k is converted into a constant input current for lIF as:

i(k) = Io + (k x Lp)

Ip = 101.2pA as a scaling factor
I0 = 2700 pA is the maximum constnt amplitude current that does not generate a spike

Weights are initialized to zero. All weights of the 8112 x 10 synapses in a fully connected layer are updated at the end of the presenation image, which lasts for time T as

w(n + 1) = w(n) + delta W

Kernals for convolution traing:

0 1 0
0 1 0
0 1 0

0 0 0
1 1 1
0 0 0

1 0 0
0 1 0
0 0 1

0 0 1
0 1 0
1 0 0

0 1 0
0 1 1
0 0 0

0 0 0
1 1 0
0 1 0

0 1 0
1 1 0
0 0 0

0 0 0
0 1 1
0 1 0

1 0 1
0 1 0
0 0 0

1 0 0
0 1 0
1 0 0

0 0 0
0 1 0
1 0 1

0 0 1
0 1 0
0 0 1
