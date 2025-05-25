import pytest
from src.network import STDPNetwork

def test_network_initialization():
    net = STDPNetwork(N_exc=10, N_inh=2)
    exc_neurons, inh_neurons, syn_exc, syn_inh = net.create_network()
    assert len(exc_neurons) == 10
    assert len(inh_neurons) == 2
    assert len(syn_exc.w) > 0
    assert len(syn_inh.w) > 0