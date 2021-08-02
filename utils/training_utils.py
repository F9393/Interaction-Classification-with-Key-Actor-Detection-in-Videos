def repeat_k_times(k):
    """
    This function returns a decorator that can be applied to 'train' function 
    to train for multiple runs. Additionally it also calculates the average metrics 
    over all the runs and returns it.

    Parameters
    ----------
    k : number of runs to train the model

    Returns
    -------
    function decorator

    """
    def decorator(train):
        def wrapper(**kwargs):
            if 'fold_no' in kwargs.keys():
                fold_no = kwargs['fold_no']
            else:
                fold_no = None
            results = []
            for i in range(k):
                best_metrics = train(CFG = kwargs['CFG'], train_set = kwargs['train_set'], valid_set = kwargs['valid_set'], save_model_subdir = kwargs['save_model_subdir'], fold_no = fold_no, run_no = i+1)
                results.append(best_metrics)
            print(f'ALL RUNS RESULT : {results}')
            avg_metrics = {key:0 for key in best_metrics.keys()}
            for run in results:
                for key,val in run.items():
                    avg_metrics[key] += val
            for key in avg_metrics:
                avg_metrics[key] /= len(results)
            print(f'AVERAGED OVER RUNS RESULT : {avg_metrics}')
            return avg_metrics
        return wrapper
    return decorator

