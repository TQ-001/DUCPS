import subprocess


def run_experiment(script_name, layers,  random, fold, epochs, tv_weight, Cauchy_weight, disable_attention= 0, skip_connection = 1):
    command = ['python', script_name, '--layers', layers, '--random', random, '--fold', fold, '--epochs', epochs, '--tv', tv_weight, '--Cauchy', Cauchy_weight, '--disable_attention', str(disable_attention), '--skip', str(skip_connection)]

    subprocess.run(command)

if __name__ == "__main__":
    Layer = [2,3,4,5,6,7,8,9]
    Random = [1, 2]
    Fold = [1, 2, 3]
    epoch = 20
    script_name = 'DUCPS.py'
    for layer in Layer:    
        for random in Random:
            for fold in Fold:
                run_experiment(script_name, str(layer),  str(random), str(fold), str(epoch), '0', '1e-2', '1', '1')

