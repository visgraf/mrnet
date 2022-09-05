import wandb


def make_run_name(run):
    config = run.config
    options = []
    sc_samples = config.get('stochastic_samples', 0)
    options.append(f'Sc{sc_samples}') 
    if config.get('stochastic_d0', False):
        options.append('D0')
    if config.get('stochastic_d1', False):
        options.append('D1')
    if config.get('stochastic_d2', False):
        options.append('D2')
    if config.get('extrema_d0', False):
        options.append('_Ex')
    if config.get('extrema_d1', False):
        options.append('D1')
    if config.get('extrema_d2', False):
        options.append('D2')
    loss_code = ''.join(config['loss_function'].split('_'))
    options.append(f"_Zr{loss_code}")
    options.append(f"{config['max_epochs_per_stage']/1000}k")
    return ''.join(options)


def delete_all(proj):
    api = wandb.Api()
    selected = api.runs(path=proj)
    print(len(selected))
    for run in selected:
        run.delete(True)

def rename_all(proj):
    api = wandb.Api()
    selected = api.runs(path=proj)
    for i, run in enumerate(selected, start=1):
        oldname = run.name
        newname = make_run_name(run)
        # if input("Pode renomear? ") == 's':
        run.name = newname
        run.update()
        print(f"{i}. Rename {oldname} to {newname}")
        
    print('Feito!')

if __name__ == '__main__':
    delete_all('siren-song/test_m1net')
    # api = wandb.Api()
    # proj = "siren-song/grayscale2D"
    # runfilter = {"config.multiresolution": "capacity",}

    # selected = api.runs(path=proj, filters=runfilter)
    # print(f"Recuperados: {len(selected)}")
    # n = int(input("Na interface: "))
    # for i, run in enumerate(selected):
    #     run.config["multiresolution"] = "pyramid"
    #     run.update()
    #     print(f"{i}. updated {run.name} to pyramid")
    # print('Done!')
    
