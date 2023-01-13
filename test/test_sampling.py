from pigp.sampling import * 

def test_uniform_random():
    # One that should pass 
    try:
        n=50
        min = [0,0]
        max = [1,1]
        sampler = UniformRandom(min,max)
        samples_internal = sampler.sample_internal(n)
        samples_boundary = sampler.sample_boundary(n)
        assert len(samples_internal) == n
        assert len(samples_boundary) == n

    except ValueError:
        print("Uniform random test should have passed!.")

def test_sobol():
    # One that should pass 
    try:
        n=32
        min = [0,0]
        max = [1,1]
        sampler = Sobol(min,max)
        samples_internal = sampler.sample_internal(n)
        samples_boundary = sampler.sample_boundary(n)
        assert len(samples_internal) == n
        assert len(samples_boundary) == n

    except ValueError:
        print("Sobol test should have passed!.")

def test_halton():
    # One that should pass 
    try:
        n=32
        min = [0,0]
        max = [1,1]
        sampler = Halton(min,max)
        samples_internal = sampler.sample_internal(n)
        samples_boundary = sampler.sample_boundary(n)
        assert len(samples_internal) == n
        assert len(samples_boundary) == n

    except ValueError:
        print("Halton test should have passed!.")

def test_latin_hypercube():
    # One that should pass 
    try:
        n=32
        min = [0,0]
        max = [1,1]
        sampler = LatinHypercube(min,max)
        samples_internal = sampler.sample_internal(n)
        samples_boundary = sampler.sample_boundary(n)
        assert len(samples_internal) == n
        assert len(samples_boundary) == n
    
    except ValueError:
        print("Latin Hypercube test should have passed!.")

if __name__ == '__main__': 
    test_uniform_random()
    test_sobol()
    test_halton()
    test_latin_hypercube