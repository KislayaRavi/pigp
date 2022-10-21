from sampling import *

def test_Sobol():
    # One that should pass 
    try:
        n=512
        min = [0,0]
        max = [1,1]
        sampler = Sobol(min,max,n)
        samples = sampler.sample()
        assert samples.size == n

    except ValueError:
        print("The test should have passed!.")

def test_Halton():
    # One that should pass 
    try:
        n=512
        min = [0,0]
        max = [1,1]
        sampler = Halton(min,max,n)
        samples = sampler.sample()
        assert samples.size == n

    except ValueError:
        print("The test should have passed!.")

def test_LatinHypercube():
    # One that should pass 
    try:
        n=512
        min = [0,0]
        max = [1,1]
        sampler = LatinHypercube(min,max,n)
        samples = sampler.sample()
        assert samples.size == n
    
    except ValueError:
        print("The test should have passed!.")

if __name__ == '__main__': 
    test_Sobol()
    test_Halton()
    test_LatinHypercube()